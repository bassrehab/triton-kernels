"""W4A16 GEMM - portable 4-bit weight-only quantized matmul (Triton, universal).

FP16 activations x INT4 weights with per-group scale/zero, dequantized inside the
kernel. Ships as a "universal" (torch-noarch) Kernel Hub build - one artifact that runs
on NVIDIA (SM80/SM90) and AMD (MI300X). The kernel lives in gemm.py (a copy of
triton_kernels/w4a16.py from https://github.com/bassrehab/triton-kernels); the submodule
is named gemm rather than w4a16 to avoid a name collision with this package.

Loading from the Hub::

    import torch
    from kernels import get_kernel

    w4a16 = get_kernel("bassrehab/w4a16", version=1, trust_remote_code=True)

    packed, scales, zeros = w4a16.quantize_weight_int4_grouped(weight_fp16, group_size=128)
    y = w4a16.w4a16_gemm(x, packed, scales, zeros, group_size=128)

``trust_remote_code=True`` is required until the publisher is on the Kernel Hub
trusted-publisher list.
"""

from .gemm import w4a16_gemm, quantize_weight_int4_grouped, pack_int4

__all__ = [
    "w4a16_gemm",
    "quantize_weight_int4_grouped",
    "pack_int4",
]
