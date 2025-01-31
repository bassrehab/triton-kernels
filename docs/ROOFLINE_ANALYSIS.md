# Roofline Analysis

Benchmark results from A100-SXM4-40GB.

## Hardware

- Memory Bandwidth: 1555 GB/s (theoretical)
- FP16: 312 TFLOPS
- INT8: 624 TOPS

## Results

![Roofline Plot](figures/roofline_all.png)

### RMSNorm

| Variant | Latency | Bandwidth | Speedup |
|---------|---------|-----------|---------|
| PyTorch RMSNorm | 0.30ms | 168 GB/s | 1.0x |
| Triton RMSNorm | 0.04ms | 1365 GB/s | **8.1x** |
| PyTorch RMSNorm+Residual | 0.32ms | 266 GB/s | 1.0x |
| Triton Fused | 0.05ms | 1285 GB/s | **6.0x** |

PyTorch launches multiple kernels and writes intermediates. Triton fuses into one kernel, hitting 88% of peak bandwidth.

### SwiGLU

| Variant | Latency | Bandwidth | Speedup |
|---------|---------|-----------|---------|
| PyTorch (F.silu + multiply) | 0.18ms | 1251 GB/s | 1.0x |
| Triton Fused | 0.11ms | 1223 GB/s | **1.6x** |

Fusion eliminates the intermediate tensor from `silu(gate)`.

### INT8 GEMM

| Batch Size | FP16 cuBLAS | INT8 Triton | INT8 cuBLAS | vs FP16 |
|------------|-------------|-------------|-------------|---------|
| 1 token | 0.091ms | 0.094ms | N/A | 0.97x |
| 32 tokens | 0.092ms | 0.094ms | 0.367ms | 0.98x |
| 2048 tokens | 0.760ms | 1.352ms | 0.732ms | **1.04x** |

See [INT8_GEMM_INVESTIGATION.md](INT8_GEMM_INVESTIGATION.md) for the full debugging story. TL;DR: original kernel was 3x slower due to using FP32 tensor cores.

Two paths now:
- `use_cublas=False`: Triton FP16 dequant, accurate, works for any batch
- `use_cublas=True`: cuBLAS INT8, faster for M>16, adds ~5% error

## Efficiency Summary

| Kernel | Bandwidth | % Peak |
|--------|-----------|--------|
| RMSNorm | 1365 GB/s | 88% |
| SwiGLU | 1223 GB/s | 79% |
| INT8 GEMM | ~480 GB/s | 31% |

RMSNorm is near-optimal. INT8 GEMM is limited by quant/dequant overhead, not memory bandwidth.

## Memory Savings

| Weight Matrix | FP16 | INT8 | Savings |
|--------------|------|------|---------|
| 4096 × 4096 | 32 MB | 17 MB | 47% |
| 11008 × 4096 | 90 MB | 45 MB | 50% |

For 7B model: ~14GB FP16 → ~7GB INT8.

## Notes

- LLM inference is memory-bound. Most ops have AI < 10 FLOPs/byte.
- Fusion helps because it reduces memory traffic, not compute.
- INT8's main benefit is 2x memory reduction. Compute speedup is marginal due to quant overhead.
- Once you hit >70% bandwidth, further gains require algorithmic changes.

## References

- https://horace.io/brrr_intro.html
- https://arxiv.org/abs/2205.14135 (FlashAttention)
- https://arxiv.org/abs/2208.07339 (LLM.int8)
