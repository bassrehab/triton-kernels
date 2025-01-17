"""
triton-kernels: High-performance GPU kernels for LLM inference using OpenAI Triton.

This package provides fused and optimized kernels for common transformer operations.
"""

__version__ = "0.1.0"

# RMSNorm kernels
from triton_kernels.rmsnorm import (
    rmsnorm,
    rmsnorm_torch,
    rmsnorm_residual_fused,
    rmsnorm_residual_torch,
    TritonRMSNorm,
)

# SwiGLU kernels
from triton_kernels.swiglu import (
    swiglu_fused,
    swiglu_torch,
    SwiGLU,
)

__all__ = [
    # RMSNorm
    "rmsnorm",
    "rmsnorm_torch",
    "rmsnorm_residual_fused",
    "rmsnorm_residual_torch",
    "TritonRMSNorm",
    # SwiGLU
    "swiglu_fused",
    "swiglu_torch",
    "SwiGLU",
]
