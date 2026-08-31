import torch
import torch.distributed as dist

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.common import get_current_device


def padding_images(images=None):
    cur_num_images = 0
    num_padding_images = 0
    image_channels = None
    image_dtype = torch.float32
    # Keep the historical CPU default when a rank has no local images; the
    # training input mover will place the resulting tensor with the rest of
    # the batch.  For an existing batch, preserve its device for concat.
    image_device = None
    if images is not None:
        cur_num_images = images.shape[0]
        if images.ndim < 2:
            raise ValueError(
                f"Expected an image batch with a channel dimension, got {tuple(images.shape)}."
            )
        image_channels = int(images.shape[1])
        if image_channels not in (3, 4):
            raise ValueError(
                "Expected RGB/RGBA images when padding, "
                f"got {image_channels} channels (shape={tuple(images.shape)})."
            )
        image_dtype = images.dtype
        image_device = images.device

    max_num_images = torch.tensor([cur_num_images], dtype=torch.long, device=get_current_device())
    dist.all_reduce(max_num_images, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.DATA))
    if max_num_images > 0:
        num_padding_images = max_num_images.item() - cur_num_images
        if num_padding_images > 0:
            image_size = gpc.config.data.force_image_size
            if image_channels is None:
                # A rank with no local images still has to contribute a
                # tensor that can be concatenated on other ranks.  New
                # layered configs expose ``model.output_channels``; retain
                # RGB as the fallback for old checkpoints/configs.
                model_config = getattr(gpc.config, "model", None)
                if isinstance(model_config, dict):
                    configured_channels = model_config.get("output_channels", 3)
                else:
                    configured_channels = getattr(model_config, "output_channels", 3)
                image_channels = int(configured_channels or 3)
                if image_channels not in (3, 4):
                    image_channels = 3
            padding_images = torch.zeros(
                (num_padding_images, image_channels, image_size, image_size),
                dtype=image_dtype,
                device=image_device,
            )
            if images is None:
                images = padding_images
            else:
                images = torch.cat((images, padding_images), dim=0)
    image_flags = [1] * cur_num_images + [0] * num_padding_images
    image_flags = torch.LongTensor(image_flags)

    return images, image_flags
