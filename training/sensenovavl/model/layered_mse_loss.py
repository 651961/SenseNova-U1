"""Loss primitives for joint RGB and layered RGBA image generation."""

import torch


def split_layered_mse(squared_error: torch.Tensor, layer_indices: torch.Tensor):
    """Return per-token RGB/alpha MSE values and base/object layer masks.

    ``squared_error`` must use pixel-major RGBA layout:
    ``[token, pixel, channel]``. Layer 0 contributes only its RGB term. Layers
    1..N contribute separate RGB and alpha terms.
    """
    if squared_error.ndim != 3 or squared_error.shape[-1] != 4:
        raise ValueError(
            "Expected elementwise squared RGBA error shaped [tokens, pixels, 4], "
            f"got {tuple(squared_error.shape)}."
        )

    layer_indices = torch.as_tensor(
        layer_indices,
        dtype=torch.long,
        device=squared_error.device,
    ).flatten()
    if layer_indices.shape[0] != squared_error.shape[0]:
        raise ValueError(
            "Layer indices must align with generated tokens: "
            f"indices={layer_indices.shape[0]}, tokens={squared_error.shape[0]}."
        )
    if (layer_indices < 0).any():
        raise ValueError("Generated tokens require non-negative layer indices.")

    rgb_mse = squared_error[..., :3].mean(dim=(1, 2))
    alpha_mse = squared_error[..., 3].mean(dim=1)
    base_mask = layer_indices == 0
    layered_mask = layer_indices > 0
    return rgb_mse, alpha_mse, base_mask, layered_mask
