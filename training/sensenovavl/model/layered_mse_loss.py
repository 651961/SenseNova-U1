"""Loss primitives for joint RGB and layered RGBA image generation."""

import torch


def _masked_mse_per_token(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average ``values`` over selected pixels, independently per token."""

    channel_count = values.shape[-1] if values.ndim == 3 else 1
    expanded_mask = mask.unsqueeze(-1) if values.ndim == 3 else mask
    numerator = (values * expanded_mask).sum(dim=tuple(range(1, values.ndim)))
    denominator = mask.sum(dim=1) * channel_count
    return numerator / denominator.clamp_min(1)


def split_layered_mse(
    squared_error: torch.Tensor,
    layer_indices: torch.Tensor,
    foreground_mask: torch.Tensor,
):
    """Split RGBA MSE into base RGB and foreground-aware object terms.

    ``squared_error`` must use pixel-major RGBA layout:
    ``[token, pixel, channel]``. Layer 0 contributes its ordinary full-image
    RGB MSE, exactly as non-layered image generation does. Layers 1..N use RGB
    supervision only where the clean target alpha is opaque. Their alpha MSE
    is split into foreground/background components so callers can balance the
    two binary classes instead of letting the usually much larger transparent
    region dominate.

    The returned foreground/background fractions are intended to multiply the
    corresponding per-token masked means before reduction. This makes the final
    denominator the number of valid pixels rather than the number of partially
    occupied image tokens.
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

    foreground_mask = torch.as_tensor(
        foreground_mask,
        dtype=torch.bool,
        device=squared_error.device,
    )
    if foreground_mask.shape != squared_error.shape[:2]:
        raise ValueError(
            "Foreground mask must align with token and pixel dimensions: "
            f"mask={tuple(foreground_mask.shape)}, "
            f"error={tuple(squared_error.shape)}."
        )

    background_mask = ~foreground_mask
    base_rgb_mse = squared_error[..., :3].mean(dim=(1, 2))
    layered_rgb_mse = _masked_mse_per_token(
        squared_error[..., :3], foreground_mask
    )
    alpha_foreground_mse = _masked_mse_per_token(
        squared_error[..., 3], foreground_mask
    )
    alpha_background_mse = _masked_mse_per_token(
        squared_error[..., 3], background_mask
    )
    foreground_fraction = foreground_mask.float().mean(dim=1)
    background_fraction = background_mask.float().mean(dim=1)
    base_mask = layer_indices == 0
    layered_mask = layer_indices > 0
    return (
        base_rgb_mse,
        layered_rgb_mse,
        alpha_foreground_mse,
        alpha_background_mse,
        base_mask,
        layered_mask,
        foreground_fraction,
        background_fraction,
    )
