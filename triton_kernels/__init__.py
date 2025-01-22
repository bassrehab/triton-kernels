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

# Quantization utilities
from triton_kernels.quantization import (
    quantize_symmetric,
    dequantize,
    quantize_weight_per_channel,
    calculate_quantization_error,
    QuantizedLinear,
)

# INT8 GEMM
from triton_kernels.quantized_matmul import (
    int8_gemm,
    int8_gemm_torch,
    Int8Linear,
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
    # Quantization
    "quantize_symmetric",
    "dequantize",
    "quantize_weight_per_channel",
    "calculate_quantization_error",
    "QuantizedLinear",
    # INT8 GEMM
    "int8_gemm",
    "int8_gemm_torch",
    "Int8Linear",
]
