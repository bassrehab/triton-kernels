# triton-kernels

High-performance GPU kernels for LLM inference, implemented in OpenAI Triton.

This repository provides educational, well-documented implementations of common transformer operations optimized for inference. Each kernel includes roofline analysis explaining *why* the optimization works at the hardware level.

## Why Custom Kernels?

LLM inference is **memory-bandwidth bound**. A 7B parameter model in FP16 requires loading 14GB of weights for every forward pass. On an A100 (2TB/s bandwidth), this takes ~7ms—while the actual computation is <0.1ms.

Custom kernels help by:
- **Fusing operations** to reduce memory round-trips
- **Quantizing weights** to reduce memory traffic
- **Maximizing bandwidth utilization** through memory-aware access patterns

## Kernels

| Kernel | Description | Speedup vs PyTorch |
|--------|-------------|-------------------|
| `rmsnorm_fused` | RMSNorm + residual add | ~1.8x |
| `swiglu_fused` | Fused SiLU-gated linear unit | ~1.4x |
| `int8_gemm` | W8A16 quantized matrix multiply | ~1.6x (2x memory) |

## Installation

```bash
pip install -e .

# Or with all dependencies
pip install -e ".[all]"
```

## Quick Start

```python
import torch
from triton_kernels import rmsnorm_fused, swiglu_fused

# Fused RMSNorm + residual
x = torch.randn(1, 2048, 4096, device='cuda', dtype=torch.float16)
residual = torch.randn_like(x)
weight = torch.ones(4096, device='cuda', dtype=torch.float16)

y = rmsnorm_fused(x, residual, weight, eps=1e-6)

# Fused SwiGLU
gate = torch.randn(1, 2048, 11008, device='cuda', dtype=torch.float16)
up = torch.randn_like(gate)

y = swiglu_fused(gate, up)
```

## Benchmarks

```bash
# Individual kernels
python -m benchmarks.bench_rmsnorm
python -m benchmarks.bench_swiglu
python -m benchmarks.bench_quantized_matmul

# Full roofline analysis
python -m benchmarks.roofline
```

## Project Structure

```
triton-kernels/
├── triton_kernels/          # Main package
│   ├── rmsnorm_fused.py
│   ├── swiglu_fused.py
│   └── quantized_matmul.py
├── benchmarks/              # Benchmark scripts
├── tests/                   # Unit tests
├── docs/                    # Documentation
└── pyproject.toml
```

## References

- [Making Deep Learning Go Brrrr (Horace He)](https://horace.io/brrr_intro.html)
- [FlashAttention (Dao et al.)](https://arxiv.org/abs/2205.14135)
- [Triton Documentation](https://triton-lang.org/)

## Author

**Subhadip Mitra** - [contact@subhadipmitra.com](mailto:contact@subhadipmitra.com)

## License

MIT