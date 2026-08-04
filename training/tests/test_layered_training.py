from copy import deepcopy
from types import SimpleNamespace

import torch
from PIL import Image

from sensenovavl.data.dataset import build_transform
from sensenovavl.data.multimodal_dataset import (
    LazySupervisedDataset,
)
from sensenovavl.data.dataset_interleaved_iterable import (
    IGNORE_TOKEN_ID,
    PackedDataset,
    build_image_gen_sequence_keep_mask,
    internevo_collate_fn,
    remap_layer_group_ids_for_packed_documents,
)
from sensenovavl.model.modules.fm_modules import ConvDecoder as TrainingConvDecoder
from sensenovavl.model.sensenovavl_moe_chat.modeling_sensenovavl_chat_mot import (
    SenseNovaVLChatMoTModel,
    build_modality_indicators,
    pack_two_branch_sequence,
)
from sensenovavl.model.sensenovavl_moe_chat.modeling_neo_vit import (
    InternVisionEmbeddings,
)
from sensenova_u1.models.neo_unify.modeling_fm_modules import (
    ConvDecoder as InferenceConvDecoder,
)
from sensenova_u1.models.neo_unify.modeling_neo_vit import NEOVisionEmbeddings


def test_rgba_pixel_head_combines_outputs_and_preserves_pretrained_rgb():
    for decoder_cls in (TrainingConvDecoder, InferenceConvDecoder):
        rgb_decoder = decoder_cls(input_dim=16, hidden_dim=16, output_channels=3)
        rgba_decoder = decoder_cls(input_dim=16, hidden_dim=16, output_channels=4)
        incompatible = rgba_decoder.load_state_dict(rgb_decoder.state_dict(), strict=True)

        assert incompatible.missing_keys == []
        assert incompatible.unexpected_keys == []
        assert rgba_decoder.conv2.out_channels == 4 * 8**2
        assert not hasattr(rgba_decoder, "alpha_conv")

        inputs = torch.randn(2, 16, 2, 3)
        rgb = rgb_decoder(inputs)
        rgba = rgba_decoder(inputs)
        assert rgba.shape == (2, 4, 64, 96)
        torch.testing.assert_close(rgba[:, :3], rgb)
        torch.testing.assert_close(rgba[:, 3], torch.zeros_like(rgba[:, 3]))


def test_combined_pixel_head_loads_legacy_split_rgba_weights():
    for decoder_cls in (TrainingConvDecoder, InferenceConvDecoder):
        rgb_decoder = decoder_cls(input_dim=16, hidden_dim=16, output_channels=3)
        legacy_state = deepcopy(rgb_decoder.state_dict())
        alpha_weight = torch.randn(64, 4, 3, 3)
        alpha_bias = torch.randn(64)
        legacy_state["alpha_conv.weight"] = alpha_weight
        legacy_state["alpha_conv.bias"] = alpha_bias

        rgba_decoder = decoder_cls(input_dim=16, hidden_dim=16, output_channels=4)
        incompatible = rgba_decoder.load_state_dict(legacy_state, strict=True)

        assert incompatible.missing_keys == []
        assert incompatible.unexpected_keys == []
        torch.testing.assert_close(
            rgba_decoder.conv2.weight[:192],
            rgb_decoder.conv2.weight,
        )
        torch.testing.assert_close(
            rgba_decoder.conv2.bias[:192],
            rgb_decoder.conv2.bias,
        )
        torch.testing.assert_close(rgba_decoder.conv2.weight[192:], alpha_weight)
        torch.testing.assert_close(rgba_decoder.conv2.bias[192:], alpha_bias)


def test_combined_pixel_head_routes_rgb_and_alpha_gradients_to_separate_rows():
    decoder = TrainingConvDecoder(input_dim=16, hidden_dim=16, output_channels=4)
    inputs = torch.randn(1, 16, 2, 2)

    decoder(inputs)[:, :3].square().mean().backward()
    assert decoder.conv2.weight.grad[:192].abs().sum() > 0
    torch.testing.assert_close(
        decoder.conv2.weight.grad[192:],
        torch.zeros_like(decoder.conv2.weight.grad[192:]),
    )

    decoder.zero_grad(set_to_none=True)
    decoder(inputs)[:, 3:].sum().backward()
    torch.testing.assert_close(
        decoder.conv2.weight.grad[:192],
        torch.zeros_like(decoder.conv2.weight.grad[:192]),
    )
    assert decoder.conv2.weight.grad[192:].abs().sum() > 0


def test_split_alpha_embedding_keeps_rgb_checkpoint_shape():
    config_kwargs = dict(
        hidden_size=8,
        llm_hidden_size=[16],
        downsample_ratio=[0.5],
        image_size=4,
        patch_size=2,
        add_pos_embedding=False,
        max_position_embeddings_vision=16,
        rope_theta_vision=10000.0,
    )
    for embedding_cls in (InternVisionEmbeddings, NEOVisionEmbeddings):
        rgb_embedding = embedding_cls(
            SimpleNamespace(
                **config_kwargs,
                num_channels=3,
                split_alpha_embedding=False,
            )
        )
        rgba_embedding = embedding_cls(
            SimpleNamespace(
                **config_kwargs,
                num_channels=4,
                split_alpha_embedding=True,
            )
        )
        incompatible = rgba_embedding.load_state_dict(
            rgb_embedding.state_dict(),
            strict=False,
        )

        assert incompatible.missing_keys == ["alpha_patch_embedding.weight"]
        assert incompatible.unexpected_keys == []
        assert rgba_embedding.patch_embedding.weight.shape[1] == 3
        torch.testing.assert_close(
            rgba_embedding.patch_embedding.weight,
            rgb_embedding.patch_embedding.weight,
        )
        torch.testing.assert_close(
            rgba_embedding.alpha_patch_embedding.weight,
            torch.zeros_like(rgba_embedding.alpha_patch_embedding.weight),
        )


def test_plain_rgba_mse_supervises_every_pixel_and_channel():
    prediction = torch.ones(2, 3, 4, requires_grad=True)
    target = torch.zeros_like(prediction)

    torch.nn.functional.mse_loss(prediction, target).backward()

    assert torch.all(prediction.grad > 0)


def test_rgba_transform_preserves_alpha_and_adds_opaque_alpha_to_rgb():
    transform = build_transform(
        is_train=False,
        input_size=2,
        resize=False,
    )
    rgba = Image.new("RGBA", (2, 2), (10, 20, 30, 64))
    rgba_tensor = transform(rgba)
    assert rgba_tensor.shape == (4, 2, 2)
    torch.testing.assert_close(
        rgba_tensor[3], torch.full((2, 2), 64 / 255), atol=1e-6, rtol=0
    )

    rgb_tensor = transform(Image.new("RGB", (2, 2), (10, 20, 30)))
    torch.testing.assert_close(rgb_tensor[3], torch.ones((2, 2)))


def test_standard_and_layered_samples_receive_default_layer_metadata():
    dataset = LazySupervisedDataset.__new__(LazySupervisedDataset)
    dataset.cfg_is_uncond_drop_independent = False
    dataset.cfg_txt_uncond_drop_prob = 0
    dataset.cfg_img_uncond_drop_prob = 0
    dataset.cfg_txtimg_uncond_drop_prob = 0
    dataset.enabel_und_loss = False

    standard = {
        "image": "output.png",
        "conversations": [
            {"from": "human", "value": "draw a cat"},
            {"from": "gpt", "value": "<image>"},
        ],
    }
    standard_output = dataset.image_gen_prepare_conv(
        deepcopy(standard), "mm_t2i", [standard["image"]]
    )
    assert standard_output[2] == [1]
    assert standard_output[5] == [0]
    assert standard_output[6] == [0]

    layered = {
        "image": ["input.png", "background.png", "object-1.png", "object-2.png"],
        "conversations": [
            {"from": "human", "value": "<image>\nseparate this image"},
            {"from": "gpt", "value": "<image><image><image>"},
        ],
    }
    layered_output = dataset.image_gen_prepare_conv(
        deepcopy(layered), "mm_it2i", deepcopy(layered["image"])
    )
    assert layered_output[2] == [0, 1, 1, 1]
    assert layered_output[5] == [-1, 0, 0, 0]
    assert layered_output[6] == [-1, 0, 1, 2]


def test_terminal_layer_sequence_matches_inference_layout():
    text, start, end, ctx = 10, 11, 12, 13
    input_ids = torch.tensor(
        [text, start, ctx, ctx, end, start, ctx, ctx, end]
    )
    keep = build_image_gen_sequence_keep_mask(
        input_ids=input_ids,
        type_ids=torch.full_like(input_ids, 3),
        data_index=torch.zeros_like(input_ids),
        image_flags=torch.ones(2, dtype=torch.long),
        image_for_gen_flags=torch.ones(2, dtype=torch.bool),
        layer_indices=torch.tensor([0, 1]),
        img_start_token_id=start,
        img_end_token_id=end,
    )
    torch.testing.assert_close(
        input_ids[keep], torch.tensor([text, start, ctx, ctx, ctx, ctx])
    )


def test_contiguous_layer_tokens_still_receive_distinct_image_ids():
    indicators = build_modality_indicators(
        torch.tensor([False, True, True, True, True]),
        image_seq_lens=[2, 2],
    )
    torch.testing.assert_close(indicators, torch.tensor([-1, 1, 1, 2, 2]))


def test_collator_keeps_rgba_layer_metadata_aligned_after_compaction():
    text, start, end, ctx = 10, 11, 12, 13
    input_ids = torch.tensor(
        [text, start, ctx, ctx, end, start, ctx, ctx, end]
    )
    feature = {
        "input_ids": input_ids,
        "labels": torch.full_like(input_ids, IGNORE_TOKEN_ID),
        "type_ids": torch.full_like(input_ids, 3),
        "data_index": torch.zeros_like(input_ids),
        "pixel_values": [torch.zeros(4, 1, 2), torch.zeros(4, 1, 2)],
        "image_flags": torch.ones(2, dtype=torch.long),
        "image_for_gen_flags": torch.ones(2, dtype=torch.bool),
        "image_for_gen_loss_flags": torch.ones(2, dtype=torch.bool),
        "is_image_duplicated_for_und_flags": torch.zeros(2, dtype=torch.bool),
        "layer_group_ids": torch.tensor([0, 0]),
        "layer_indices": torch.tensor([0, 1]),
        "worker_state_key": "test",
        "worker_state_dict": b"",
    }

    batch, _ = internevo_collate_fn(
        [feature],
        max_item_length=9,
        img_start_token_id=start,
        img_token_id=ctx,
        img_end_token_id=end,
        ignored_token_ids=[],
        micro_num=1,
        len2weight=lambda _: 1.0,
        patch_size=1,
    )

    torch.testing.assert_close(
        batch["input_ids"][0, :6],
        torch.tensor([text, start, ctx, ctx, ctx, ctx]),
    )
    assert batch["images"][0].shape == (4, 4)
    torch.testing.assert_close(
        batch["image_grid_hw"][0], torch.tensor([[1, 2], [1, 2]])
    )
    torch.testing.assert_close(batch["layer_group_ids"][0], torch.tensor([0, 0]))
    torch.testing.assert_close(batch["layer_indices"][0], torch.tensor([0, 1]))


def test_layer_groups_are_unique_across_packed_documents():
    remapped, next_group = remap_layer_group_ids_for_packed_documents(
        input_ids=torch.tensor(
            [11, 13, 12, 11, 13, 12, 11, 13, 12, 11, 13, 12]
        ),
        data_index=torch.tensor([0] * 6 + [1] * 6),
        image_flags=torch.ones(4, dtype=torch.long),
        layer_group_ids=torch.tensor([0, 0, 0, 0]),
        img_start_token_id=11,
        next_global_layer_group_id=0,
    )
    torch.testing.assert_close(remapped, torch.tensor([0, 0, 1, 1]))
    assert next_group == 2


def test_packed_split_keeps_a_layer_group_in_one_document():
    text, start, end, ctx = 10, 11, 12, 13
    first_document = torch.tensor([text, start, ctx, end, text])
    layered_document = torch.tensor(
        [text, start, ctx, ctx, end, start, ctx, ctx, end]
    )
    input_ids = torch.cat((first_document, layered_document))
    buffer = {
        "input_ids": input_ids,
        "labels": torch.zeros_like(input_ids),
        "type_ids": torch.tensor(
            [0] * len(first_document) + [4] * len(layered_document)
        ),
        "data_index": torch.tensor(
            [0] * len(first_document) + [1] * len(layered_document)
        ),
        "pixel_values": [torch.zeros(1), torch.zeros(1), torch.zeros(1)],
        "image_flags": torch.ones(3, dtype=torch.long),
        "image_for_gen_flags": torch.tensor([False, True, True]),
        "image_for_gen_loss_flags": torch.tensor([False, True, True]),
        "is_image_duplicated_for_und_flags": torch.zeros(3, dtype=torch.bool),
        "layer_group_ids": torch.tensor([-1, 0, 0]),
        "layer_indices": torch.tensor([-1, 0, 1]),
    }

    parts = PackedDataset.split_buffer(
        buffer,
        max_tokens=10,
        img_start_token_id=start,
        img_token_id=ctx,
        img_end_token_id=end,
    )

    assert [len(part["input_ids"]) for part in parts] == [5, 9]
    torch.testing.assert_close(parts[1]["layer_group_ids"], torch.tensor([0, 0]))
    torch.testing.assert_close(parts[1]["layer_indices"], torch.tensor([0, 1]))


def test_packed_layer_positions_match_layered_inference_positions():
    hidden = torch.zeros(1, 6, 2)
    indexes = torch.tensor(
        [[0, 0, 0], [10, 0, 0], [10, 0, 1], [14, 0, 0], [14, 0, 1], [15, 0, 0]]
    )
    outputs = pack_two_branch_sequence(
        hidden_states=hidden,
        indexes=indexes,
        document_ids=torch.zeros(6, dtype=torch.int32),
        modality_indicators=torch.tensor([-1, 1, 1, 2, 2, -1]),
        image_gen_indicators=torch.tensor([False, True, True, True, True, False]),
        layer_group_indicators=torch.tensor([-1, 0, 0, 0, 0, -1]),
        layer_index_indicators=torch.tensor([0, 0, 0, 1, 1, 0]),
        dup_boundary=torch.zeros(6, dtype=torch.bool),
    )
    packed_indexes = outputs[1]
    token_positions = outputs[-2]
    torch.testing.assert_close(
        packed_indexes[:4, 0], torch.tensor([10, 10, 11, 11])
    )
    torch.testing.assert_close(
        token_positions[:4], torch.tensor([10, 10, 14, 14])
    )


def test_target_builder_shares_timestep_without_fixing_first_alpha(monkeypatch):
    model = SenseNovaVLChatMoTModel.__new__(SenseNovaVLChatMoTModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=1))
    model.patch_size = 1
    model.downsample_ratio = 1.0
    model.noise_scale = 1.0
    model.noise_scale_mode = "fixed"
    model.noise_scale_base_image_seq_len = 1
    model.noise_scale_max_value = 8.0
    model.P_std = 0.0
    model.P_mean = 0.0
    model.time_schedule = "standard"
    model.timestep_shift = 1.0
    model.tp_world_size = 1
    model.t_eps = 0.05
    monkeypatch.setattr(
        torch,
        "randn_like",
        lambda value: torch.full_like(value, 0.25),
    )

    # RGB is in ImageNet-normalized understanding space; alpha remains [0, 1].
    pixels = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    outputs = model.prepare_image_gen_targets(
        pixels,
        image_for_gen_flags=torch.tensor([True, True]),
        grid_hw=torch.tensor([[1, 2], [1, 2]]),
        layer_group_ids=torch.tensor([0, 0]),
        layer_indices=torch.tensor([0, 1]),
    )
    (
        _,
        image_gen_z,
        image_gen_v,
        image_gen_t,
        _,
        _,
    ) = outputs

    torch.testing.assert_close(image_gen_t, torch.full((4,), 0.5))
    torch.testing.assert_close(
        image_gen_z[:, 3], torch.tensor([0.625, 0.625, -0.375, 0.625])
    )
    assert torch.all(image_gen_v[:2, 3] != 0)
    clean_rgba = image_gen_z + (1 - image_gen_t[:, None]) * image_gen_v
    torch.testing.assert_close(
        clean_rgba[:, 3], torch.tensor([1.0, 1.0, -1.0, 1.0])
    )
