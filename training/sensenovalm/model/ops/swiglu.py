# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Portions derived from Liger Kernel's SwiGLU implementation.
# Copyright 2024 LinkedIn Corporation. Licensed under BSD-2-Clause.
"""Stride-aware SwiGLU for a fused gate/up projection."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from liger_kernel.ops.utils import calculate_settings


@triton.jit
def _fused_gate_up_swiglu_forward_kernel(
    gate_up_ptr,
    output_ptr,
    gate_up_row_stride,
    output_row_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    gate_up_row = gate_up_ptr + row_idx * gate_up_row_stride
    output_row = output_ptr + row_idx * output_row_stride

    gate = tl.load(gate_up_row + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(gate_up_row + n_cols + offsets, mask=mask, other=0)
    silu_gate = gate * tl.sigmoid(gate)
    output = silu_gate.cast(up.dtype) * up
    tl.store(output_row + offsets, output, mask=mask)


@triton.jit
def _fused_gate_up_swiglu_backward_kernel(
    grad_output_ptr,
    gate_up_ptr,
    grad_output_row_stride,
    gate_up_row_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    grad_output_row = grad_output_ptr + row_idx * grad_output_row_stride
    gate_up_row = gate_up_ptr + row_idx * gate_up_row_stride

    grad_output = tl.load(grad_output_row + offsets, mask=mask, other=0)
    gate = tl.load(gate_up_row + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(gate_up_row + n_cols + offsets, mask=mask, other=0)

    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    grad_up = grad_output * silu_gate
    grad_gate = grad_output * (silu_gate * (1 - sigmoid_gate) + sigmoid_gate) * up

    # Reuse the saved projection output for its gradient, as Liger's two-input
    # SwiGLU backward does. The two halves form one contiguous Linear gradient.
    tl.store(gate_up_row + offsets, grad_gate, mask=mask)
    tl.store(gate_up_row + n_cols + offsets, grad_up, mask=mask)


class _FusedGateUpSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up: torch.Tensor) -> torch.Tensor:
        input_shape = gate_up.shape
        intermediate_size = input_shape[-1] // 2
        gate_up_2d = gate_up.view(-1, 2 * intermediate_size)
        output_2d = torch.empty(
            (gate_up_2d.shape[0], intermediate_size),
            dtype=gate_up.dtype,
            device=gate_up.device,
        )

        block_size, num_warps = calculate_settings(intermediate_size)
        _fused_gate_up_swiglu_forward_kernel[(gate_up_2d.shape[0],)](
            gate_up_2d,
            output_2d,
            gate_up_2d.stride(0),
            output_2d.stride(0),
            n_cols=intermediate_size,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

        ctx.input_shape = input_shape
        ctx.intermediate_size = intermediate_size
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        ctx.save_for_backward(gate_up_2d)
        return output_2d.view(*input_shape[:-1], intermediate_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (gate_up_2d,) = ctx.saved_tensors
        grad_output_2d = grad_output.contiguous().view(-1, ctx.intermediate_size)

        _fused_gate_up_swiglu_backward_kernel[(gate_up_2d.shape[0],)](
            grad_output_2d,
            gate_up_2d,
            grad_output_2d.stride(0),
            gate_up_2d.stride(0),
            n_cols=ctx.intermediate_size,
            BLOCK_SIZE=ctx.block_size,
            num_warps=ctx.num_warps,
        )

        return gate_up_2d.view(ctx.input_shape)


def fused_gate_up_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """Compute ``silu(gate) * up`` directly from ``[..., gate || up]``.

    The fused Linear output is contiguous, while either half obtained through
    ``split`` is not. Reading both halves in one kernel avoids the two copies
    that Liger's generic contiguous-input wrapper would otherwise create.
    """
    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(f"The fused gate/up dimension must be even, got {gate_up.shape[-1]}.")

    intermediate_size = gate_up.shape[-1] // 2
    if gate_up.numel() == 0 or gate_up.device.type != "cuda":
        gate, up = gate_up.split(intermediate_size, dim=-1)
        return F.silu(gate) * up

    if not gate_up.is_contiguous():
        gate_up = gate_up.contiguous()
    return _FusedGateUpSwiGLUFunction.apply(gate_up)


__all__ = ["fused_gate_up_swiglu"]
