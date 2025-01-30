# Roofline Analysis

## Hardware Configuration

- **GPU**: NVIDIA A100-SXM4-40GB
- **Compute Capability**: 8.0
- **Memory Bandwidth**: 1555 GB/s (theoretical peak)
- **FP16 Peak**: 312 TFLOPS (theoretical peak)

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

| Variant | Status |
|---------|--------|
| Basic RMSNorm | Memory-bound (AI ≈ 0.7) |
| Fused RMSNorm + Residual | Memory-bound (AI ≈ 0.6) |

**Why fusion helps**: RMSNorm reads the input tensor and writes the output.
When fused with residual addition, we avoid materializing the intermediate
`x + residual` tensor, saving one full memory round-trip.

**Speedup**: ~1.5-2x for fused variant

### SwiGLU

| Variant | Status |
|---------|--------|
| PyTorch (F.silu + multiply) | Memory-bound (AI ≈ 0.4) |
| Triton Fused | Memory-bound (AI ≈ 0.7) |

**Why fusion helps**: The separate `F.silu(gate) * up` operation writes the
intermediate `silu(gate)` to memory, then reads it back. Fusion eliminates
this intermediate tensor.

**Speedup**: ~1.3-1.5x

### INT8 GEMM

| Variant | Status |
|---------|--------|
| FP16 GEMM | Memory-bound for small M, compute-bound for large M |
| INT8 GEMM | Same, but 2x less weight memory traffic |

**Why quantization helps**: For memory-bound GEMMs (small batch/sequence),
the weight matrix dominates memory traffic. INT8 weights are 2x smaller than
FP16, directly reducing memory traffic.

**Speedup**: ~1.5-2x for memory-bound cases (seq_len < 256)

## Key Insights

### 1. LLM Inference is Memory-Bound

Most operations in LLM inference have low arithmetic intensity (<10 FLOPs/byte),
making them memory-bound. This is why:
- Custom kernels focus on reducing memory traffic (fusion, quantization)
- Raw FLOPS don't matter as much as memory bandwidth utilization
- The same operation can be 10x faster just by using memory more efficiently

### 2. Fusion is Free Performance

Kernel fusion eliminates intermediate tensors without adding computational cost.
For memory-bound operations, this directly translates to speedup proportional
to the memory traffic reduction.

### 3. Quantization Trades Accuracy for Speed

INT8 quantization reduces memory traffic by 2x but introduces quantization error.
For LLM weights, this error is typically small (<1% relative) and doesn't
significantly affect model quality.

### 4. The Roofline Ceiling

No kernel can exceed the roofline. When a kernel achieves >70% of theoretical
peak bandwidth, further optimization requires algorithmic changes (different
memory access patterns, not just code tuning).

## Achieved Efficiency

| Kernel Category | Achieved Bandwidth | % of Peak |
|-----------------|-------------------|-----------|
| RMSNorm (Triton) | ~70-80% | Good |
| SwiGLU (Triton) | ~60-70% | Good |
| INT8 GEMM | ~50-70% | Acceptable |

Note: These numbers are GPU-dependent. Higher efficiency is typically seen on
newer architectures with better memory systems.

## Recommendations for LLM Inference

1. **Always fuse normalization with residual add** - Free 1.5-2x speedup
2. **Quantize weights to INT8** for memory-bound GEMMs (decode phase)
3. **Use FP16 for compute-bound** operations (prefill with long contexts)
4. **Profile your specific workload** - The optimal strategy depends on
   batch size, sequence length, and model architecture

## References

- [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) - Essential reading on GPU optimization
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135) - IO-aware algorithm design
- [LLM.int8() Paper](https://arxiv.org/abs/2208.07339) - INT8 quantization for LLMs
- [Triton Documentation](https://triton-lang.org/) - Triton programming guide
