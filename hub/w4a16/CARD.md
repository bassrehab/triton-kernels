---
license: mit
tags:
- kernel
- triton
- quantization
- gptq
- awq
- llm-inference
---

# W4A16 GEMM (Triton, universal)

A cross-platform, Triton-native **4-bit weight-only GEMM**: FP16 activations times INT4
weights (GPTQ / AWQ style, group-wise scales), dequantized inside the kernel. The fast
W4A16 kernels (Marlin, AWQ, exllama) are CUDA-only; this one is pure Triton and runs on
NVIDIA (SM80/SM90) and AMD (MI300X) from a single universal build.

```
y = x @ dequant(W_q)      # x FP16 (M,K), W_q INT4 packed (K//8, N), group-wise scale/zero
```

It **beats cuBLAS FP16 in the decode regime** (~1.05-1.17x at batch size 1 on large FFN
shapes, via a split-K dispatch) and matches FP16 for larger batches, with a 4x
weight-memory reduction.

## Usage

```python
import torch
from kernels import get_kernel

# trust_remote_code=True is required until the publisher is on the trusted list.
w4a16 = get_kernel("bassrehab/w4a16", version=1, trust_remote_code=True)

K, N, group_size = 4096, 14336, 128
weight = torch.randn(K, N)                    # FP16-range weights to quantize (CPU ok)
packed, scales, zeros = w4a16.quantize_weight_int4_grouped(weight, group_size)

x = torch.randn(8, K, dtype=torch.float16, device="cuda")
packed = packed.cuda(); scales = scales.cuda().half(); zeros = zeros.cuda().half()
y = w4a16.w4a16_gemm(x, packed, scales, zeros, group_size)   # (8, N) FP16
```

## Public API

| Symbol | Purpose |
|--------|---------|
| `w4a16_gemm(x, packed, scales, zeros, group_size)` | 4-bit weight-only GEMM. Auto-dispatches split-K for decode. |
| `quantize_weight_int4_grouped(weight, group_size, symmetric)` | Quantize + pack an FP weight `(K, N)`. |
| `pack_int4(q)` | Pack an `(K, N)` tensor of values in `[0,15]` into `(K//8, N)` int32. |

## Weight format

Values in `[0, 15]`, 8 nibbles packed per int32 along K -> packed `(K//8, N)`; one scale
and one zero-point per group of `group_size` rows along K (64 or 128; `group_size` must
be a multiple of 64). Dequant: `w = (q - zero) * scale`. Covers asymmetric (per-group
zero) and symmetric (fixed zero 8) quantization.

## Supported hardware

| Backend | Status |
|---------|--------|
| NVIDIA A100 / H100 (SM80/SM90) | Primary; correctness + benchmarks |
| AMD MI300X (ROCm via Triton) | Pending correctness validation |

## Status

**Version 1.** Load with `version=1`. Correctness validated on A100 (30/30). The full
memory-bandwidth roofline win needs a Marlin-style layout (future work). Source, roofline
analysis, and writeup: <https://github.com/bassrehab/triton-kernels>.
