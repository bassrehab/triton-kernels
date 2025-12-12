# Roofline Analysis

## Hardware Configuration

- **GPU**: NVIDIA A100-SXM4-40GB
- **Compute Capability**: 8.0
- **Memory Bandwidth**: 1555 GB/s (theoretical peak)
- **FP16 Peak**: 312 TFLOPS (theoretical peak)
- **INT8 Peak**: 624 TOPS (theoretical peak)

## Summary

This analysis benchmarks custom Triton kernels against PyTorch baselines for
common LLM inference operations. All operations are tested with LLaMA 7B-style
dimensions (hidden_dim=4096, ffn_dim=11008).

## Roofline Model

![Roofline Plot](figures/roofline_all.png)

The roofline model visualizes the relationship between:
- **Arithmetic Intensity (AI)**: FLOPs per byte of memory traffic
- **Achieved Performance**: GFLOPS

Operations below the diagonal "roofline" are **memory-bound** (limited by memory
bandwidth), while operations on the plateau are **compute-bound** (limited by
peak FLOPS).

## Kernel Analysis

### RMSNorm

| Variant | Latency | Bandwidth | Speedup |
|---------|---------|-----------|---------|
| PyTorch RMSNorm | 0.30ms | 168 GB/s | 1.0x |
| Triton RMSNorm | 0.04ms | 1365 GB/s | **8.1x** |
| PyTorch RMSNorm+Residual | 0.32ms | 266 GB/s | 1.0x |
| Triton RMSNorm+Residual (Fused) | 0.05ms | 1285 GB/s | **6.0x** |

**Why Triton is 8x faster**: PyTorch RMSNorm launches multiple small kernels and
materializes intermediate tensors. Triton fuses everything into one kernel, achieving
88% of peak memory bandwidth (1365/1555 GB/s).

### SwiGLU

| Variant | Latency | Bandwidth | Speedup |
|---------|---------|-----------|---------|
| PyTorch (F.silu + multiply) | 0.18ms | 1251 GB/s | 1.0x |
| Triton Fused | 0.11ms | 1223 GB/s | **1.6x** |

**Why fusion helps**: The separate `F.silu(gate) * up` operation writes the
intermediate `silu(gate)` to memory, then reads it back. Fusion eliminates
this intermediate tensor. Both achieve ~80% of peak bandwidth.

### INT8 GEMM (W8A16)

This is the most complex kernel with multiple implementation paths:

| Batch Size | FP16 cuBLAS | INT8 Triton | INT8 cuBLAS | Best INT8 vs FP16 |
|------------|-------------|-------------|-------------|-------------------|
| 1 token | 0.091ms | 0.094ms | N/A | 0.97x (Triton) |
| 32 tokens | 0.092ms | 0.094ms | 0.367ms | 0.98x (Triton) |
| 2048 tokens | 0.760ms | 1.352ms | 0.732ms | **1.04x** (cuBLAS) |

**Key findings from our investigation:**

1. **Original bug**: INT8 kernel was 3x SLOWER because it used FP32 tensor cores
   instead of FP16 tensor cores (fixed in this branch).

2. **Three INT8 paths available**:
   - **Triton FP16 dequant** (`use_cublas=False`): Dequantizes INT8 weights to FP16
     in registers, uses FP16 tensor cores. Accurate, works for any batch size.
   - **cuBLAS INT8** (`use_cublas=True`): Quantizes activations on-the-fly, uses
     INT8 tensor cores. Faster for M>16 but adds ~3-8% error.
   - **Fused Triton INT8** (`_int8_gemm_fused`): Fuses quant/dequant into kernel.
     Experimental, not as fast as cuBLAS.

3. **Why INT8 doesn't give 2x speedup**:
   - INT8 tensor cores are 2x faster than FP16 (624 vs 312 TFLOPS)
   - BUT: We need to quantize FP16 activations → INT8 (~0.19ms overhead)
   - AND: Dequantize INT32 output → FP16 (~0.37ms overhead)
   - This overhead cancels the compute speedup
   - **Main value of INT8 is 2x memory savings**, not compute speedup

4. **When to use which path**:
   - **Small batches (M≤16)**: Use default Triton path (accurate, no cuBLAS option)
   - **Large batches (M>16)**: Use `use_cublas=True` if speed > accuracy

## Achieved Efficiency

| Kernel Category | Achieved Bandwidth | % of Peak | Status |
|-----------------|-------------------|-----------|--------|
| RMSNorm (Triton) | 1365 GB/s | 88% | Excellent |
| SwiGLU (Triton) | 1223 GB/s | 79% | Good |
| INT8 GEMM (Triton) | ~480 GB/s | 31% | Memory-bound |
| INT8 GEMM (cuBLAS) | 146 GB/s | 9% | Compute-bound |

Note: INT8 GEMM bandwidth is lower because it's compute-bound for large batches.

## Memory Savings

| Weight Matrix | FP16 Size | INT8 Size | Savings |
|--------------|-----------|-----------|---------|
| 4096 × 4096 | 32 MB | 17 MB | 47% |
| 11008 × 4096 | 90 MB | 45 MB | 50% |
| 32000 × 4096 | 262 MB | 131 MB | 50% |

For a 7B parameter model, INT8 weights reduce memory from ~14GB to ~7GB.

## Key Insights

### 1. LLM Inference is Memory-Bound

Most operations in LLM inference have low arithmetic intensity (<10 FLOPs/byte),
making them memory-bound. This is why:
- Custom kernels focus on reducing memory traffic (fusion, quantization)
- Raw FLOPS don't matter as much as memory bandwidth utilization
- RMSNorm achieves 88% of peak bandwidth - near optimal

### 2. Fusion is Free Performance

Kernel fusion eliminates intermediate tensors without adding computational cost.
For memory-bound operations, this directly translates to speedup proportional
to the memory traffic reduction.

### 3. INT8 Quantization: Memory vs Compute Tradeoff

INT8 quantization reduces memory traffic by 2x but:
- For compute-bound operations (large batches), quantization overhead dominates
- For memory-bound operations (small batches), memory savings help
- The "sweet spot" is inference with batch_size=1, seq_len < 256

### 4. The Roofline Ceiling

No kernel can exceed the roofline. When a kernel achieves >70% of theoretical
peak bandwidth, further optimization requires algorithmic changes (different
memory access patterns, not just code tuning).

## Recommendations for LLM Inference

1. **Always fuse normalization with residual add** - Free 6-8x speedup
2. **Use Triton SwiGLU fusion** - Free 1.6x speedup
3. **For weights, INT8 is recommended** for 2x memory reduction
4. **For INT8 GEMM speed**, use `use_cublas=True` for large batches (M>16)
5. **Profile your specific workload** - The optimal strategy depends on
   batch size, sequence length, and model architecture

## Related Documentation

- [INT8 GEMM Investigation](INT8_GEMM_INVESTIGATION.md) - Detailed write-up of the INT8 performance investigation

## References

- [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) - Essential reading on GPU optimization
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135) - IO-aware algorithm design
- [LLM.int8() Paper](https://arxiv.org/abs/2208.07339) - INT8 quantization for LLMs
- [Triton Documentation](https://triton-lang.org/) - Triton programming guide
