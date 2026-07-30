# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
"""RMSNorm backend used by SenseNovaLM training."""

from liger_kernel.transformers.rms_norm import LigerRMSNorm

RMSNorm = LigerRMSNorm

__all__ = ["RMSNorm"]
