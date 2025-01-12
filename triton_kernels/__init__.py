"""
triton-kernels: High-performance GPU kernels for LLM inference using OpenAI Triton.

This package provides fused and optimized kernels for common transformer operations.
"""

__version__ = "0.1.0"

# RMSNorm kernels
from triton_kernels.rmsnorm import rmsnorm, rmsnorm_torch, TritonRMSNorm

__all__ = [
    "rmsnorm",
    "rmsnorm_torch",
    "TritonRMSNorm",
]
