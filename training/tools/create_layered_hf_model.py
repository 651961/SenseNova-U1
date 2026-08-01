#!/usr/bin/env python3
"""Create a combined-RGBA Pixel Head model from an RGB HuggingFace model.

Unchanged top-level files are hard-linked from the source directory. The two
small shards containing the generation input embedding and Pixel Head are
rewritten so the resulting directory is a complete model without duplicating
the multi-gigabyte LLM shards.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


RGB_CHANNELS = 3
RGBA_CHANNELS = 4
FINAL_UPSCALE = 8
RGB_SUBPIXEL_CHANNELS = RGB_CHANNELS * FINAL_UPSCALE**2
RGBA_SUBPIXEL_CHANNELS = RGBA_CHANNELS * FINAL_UPSCALE**2

PIXEL_HEAD_WEIGHT = "fm_modules.fm_head.conv2.weight"
PIXEL_HEAD_BIAS = "fm_modules.fm_head.conv2.bias"
RGB_PATCH_WEIGHT = (
    "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight"
)
ALPHA_PATCH_WEIGHT = (
    "fm_modules.vision_model_mot_gen.embeddings.alpha_patch_embedding.weight"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand a SenseNova U1 RGB HF checkpoint to combined RGBA."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _copy_top_level_files_as_hardlinks(source: Path, output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise ValueError(f"Output directory must be empty: {output}")
    else:
        output.mkdir(parents=True)

    for source_file in source.iterdir():
        if source_file.is_file():
            os.link(source_file, output / source_file.name)


def _rewrite_safetensors(
    shard_path: Path,
    transform,
) -> None:
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    original_mode = stat.S_IMODE(shard_path.stat().st_mode)
    tensors = load_file(shard_path, device="cpu")
    transform(tensors)

    temporary_path = shard_path.with_name(f".{shard_path.name}.tmp")
    save_file(tensors, temporary_path, metadata=metadata)
    os.chmod(temporary_path, original_mode)
    os.replace(temporary_path, shard_path)


def _expand_pixel_head(tensors: dict[str, torch.Tensor]) -> None:
    weight = tensors[PIXEL_HEAD_WEIGHT]
    bias = tensors[PIXEL_HEAD_BIAS]
    expected_weight_shape = (RGB_SUBPIXEL_CHANNELS, 256, 3, 3)
    if tuple(weight.shape) != expected_weight_shape:
        raise ValueError(
            f"Expected RGB Pixel Head {PIXEL_HEAD_WEIGHT} shaped "
            f"{expected_weight_shape}, got {tuple(weight.shape)}"
        )
    if tuple(bias.shape) != (RGB_SUBPIXEL_CHANNELS,):
        raise ValueError(
            f"Expected RGB Pixel Head bias shaped {(RGB_SUBPIXEL_CHANNELS,)}, "
            f"got {tuple(bias.shape)}"
        )

    expanded_weight = weight.new_zeros(
        RGBA_SUBPIXEL_CHANNELS,
        *weight.shape[1:],
    )
    expanded_bias = bias.new_zeros(RGBA_SUBPIXEL_CHANNELS)
    expanded_weight[:RGB_SUBPIXEL_CHANNELS].copy_(weight)
    expanded_bias[:RGB_SUBPIXEL_CHANNELS].copy_(bias)
    tensors[PIXEL_HEAD_WEIGHT] = expanded_weight
    tensors[PIXEL_HEAD_BIAS] = expanded_bias


def _add_alpha_input_embedding(tensors: dict[str, torch.Tensor]) -> None:
    rgb_weight = tensors[RGB_PATCH_WEIGHT]
    if rgb_weight.ndim != 4 or rgb_weight.shape[1] != RGB_CHANNELS:
        raise ValueError(
            f"Expected RGB patch embedding [out, 3, kh, kw], got "
            f"{tuple(rgb_weight.shape)}"
        )
    tensors[ALPHA_PATCH_WEIGHT] = rgb_weight.new_zeros(
        rgb_weight.shape[0],
        1,
        rgb_weight.shape[2],
        rgb_weight.shape[3],
    )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("Source and output directories must differ.")
    if not (source / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"Missing source model index: {source}")

    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    pixel_head_shard = weight_map[PIXEL_HEAD_WEIGHT]
    patch_embedding_shard = weight_map[RGB_PATCH_WEIGHT]

    _copy_top_level_files_as_hardlinks(source, output)
    _rewrite_safetensors(output / pixel_head_shard, _expand_pixel_head)
    _rewrite_safetensors(output / patch_embedding_shard, _add_alpha_input_embedding)

    weight_map[ALPHA_PATCH_WEIGHT] = patch_embedding_shard
    index_path = output / "model.safetensors.index.json"
    temporary_index = output / ".model.safetensors.index.json.tmp"
    temporary_index.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    os.chmod(temporary_index, stat.S_IMODE(index_path.stat().st_mode))
    os.replace(temporary_index, index_path)

    print(f"Created combined-RGBA model: {output}")
    print(
        f"Pixel Head: {tuple((RGBA_SUBPIXEL_CHANNELS, 256, 3, 3))}; "
        f"RGB rows copied={RGB_SUBPIXEL_CHANNELS}, Alpha rows zeroed="
        f"{RGBA_SUBPIXEL_CHANNELS - RGB_SUBPIXEL_CHANNELS}"
    )


if __name__ == "__main__":
    main()
