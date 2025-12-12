# INT8 GEMM Performance Investigation

This document captures the investigation and learnings from optimizing the INT8 GEMM kernel on A100 GPUs.

## Table of Contents

1. [The Problem](#the-problem)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Solution Approaches](#solution-approaches)
4. [Performance Profiling](#performance-profiling)
5. [Final Implementation](#final-implementation)
6. [Key Learnings](#key-learnings)

---

## The Problem

### Initial Observation

When benchmarking the INT8 GEMM kernel against FP16 cuBLAS, we observed:

```
FP16 GEMM: 0.76ms, 242 TFLOPS
INT8 GEMM: 2.43ms,  76 TFLOPS  ← 0.31x speedup (3x SLOWER!)
```

This was completely backwards. INT8 tensor cores on A100 should deliver **2x** the throughput of FP16 (624 TOPS vs 312 TFLOPS).

### Expected vs Actual

| Metric | Expected | Actual |
|--------|----------|--------|
| INT8 vs FP16 speedup | 1.5-2x | 0.31x (3x slower) |
| INT8 TFLOPS | ~400-500 | 76 |
| Memory bandwidth benefit | 2x from INT8 weights | Negligible |

---

## Root Cause Analysis

### Bug #1: Using FP32 Tensor Cores

The original kernel code:

```python
# WRONG: This uses FP32 tensor cores (or CUDA cores)
b_fp32 = b_int8.to(tl.float32)
acc += tl.dot(a.to(tl.float32), b_fp32)
```

**Problem**: Converting both operands to FP32 before `tl.dot` causes Triton to use FP32 tensor cores (or fall back to CUDA cores). A100 tensor cores support:
- FP16 × FP16 → FP32 (312 TFLOPS)
- BF16 × BF16 → FP32 (312 TFLOPS)
- INT8 × INT8 → INT32 (624 TOPS)
- **NOT** FP32 × FP32 for matmul

### Bug #2: No Autotuning

The kernel used fixed block sizes:
```python
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
```

These were not optimized for A100's memory hierarchy and SM count.

### Bug #3: Transpose on Every Call

```python
weight_t = weight_int8.t().contiguous()  # Called every forward pass!
```

This created a new tensor copy on every call, adding overhead.

---

## Solution Approaches

We explored three different approaches:

### Approach 1: FP16 Tensor Cores with INT8 Dequantization

**Idea**: Load INT8 weights, dequantize to FP16 in registers, use FP16 tensor cores.

```python
# Load INT8 weights
b_int8 = tl.load(b_ptrs, mask=b_mask, other=0)

# Dequantize to FP16 in registers (essentially free)
b_fp16 = b_int8.to(tl.float16)

# Use FP16 tensor cores
acc += tl.dot(a, b_fp16, out_dtype=tl.float32)
```

**Result**: Matched FP16 performance (~0.76ms), but no speedup.

**Why it works**:
- FP16 tensor cores are triggered
- INT8→FP16 conversion happens in registers (fast)
- Memory traffic still reduced by 2x for weights

### Approach 2: cuBLAS INT8 via torch._int_mm

**Idea**: Use PyTorch's native INT8 matmul which calls cuBLAS.

```python
# Quantize activations on-the-fly
x_int8 = (x * scale).round().clamp(-128, 127).to(torch.int8)

# INT8 matmul using cuBLAS tensor cores
y_int32 = torch._int_mm(x_int8, weight_int8.t())

# Dequantize output
y_fp16 = (y_int32.float() * combined_scale).half()
```

**Result**:
- INT8 matmul alone: 0.45ms (413 TOPS, 66% of peak)
- Full pipeline: 0.73ms (1.04x speedup vs FP16)

**Tradeoff**: Adds 3-8% numerical error from activation quantization.

### Approach 3: Fused Triton INT8 Kernel

**Idea**: Fuse activation quantization, INT8 matmul, and dequantization into one kernel.

```python
@triton.jit
def _int8_gemm_fused_kernel(...):
    # Load FP16 activations
    a_fp16 = tl.load(a_ptrs, ...)

    # Quantize to INT8 in registers
    a_int8 = (a_fp16 * inv_scale).to(tl.int8)

    # Load INT8 weights
    b_int8 = tl.load(b_ptrs, ...)

    # INT8 matmul
    acc += tl.dot(a_int8, b_int8, out_dtype=tl.int32)

    # Dequantize and store
    tl.store(c_ptrs, (acc * combined_scale).to(tl.float16))
```

**Result**: 1.2ms - slower than cuBLAS due to Triton's INT8 codegen.

---

## Performance Profiling

### Overhead Breakdown (M=2048, K=4096, N=11008)

```
Component               Time (ms)   % of Total
─────────────────────────────────────────────
absmax computation      0.053       5%
Activation quantization 0.133       12%
INT8 matmul (_int_mm)   0.447       42%
Dequantization          0.367       34%
Other overhead          0.097       9%
─────────────────────────────────────────────
Total INT8 pipeline     1.097       100%
FP16 baseline           0.912       (for comparison)
```

**Key insight**: The INT8 matmul itself is 2x faster (0.45ms vs ~0.9ms), but quantization/dequantization overhead (0.55ms) cancels the benefit.

### cuBLAS INT8 Layout Discovery

Critical finding about `torch._int_mm`:

```python
# SLOW (2.7ms): B is row-major contiguous
y = torch._int_mm(A, B_contiguous)

# FAST (0.5ms): B is column-major (transpose view)
y = torch._int_mm(A, B.t())  # Don't call .contiguous()!
```

cuBLAS INT8 GEMM is optimized for column-major B matrix. Passing a transpose **view** (not contiguous) triggers the fast path.

### Batch Size Impact

| Batch Size (M) | FP16 | INT8 Triton | INT8 cuBLAS | Winner |
|----------------|------|-------------|-------------|--------|
| 1 | 0.09ms | 0.09ms | N/A | Tie |
| 32 | 0.09ms | 0.09ms | 0.51ms | Triton |
| 128 | 0.11ms | 0.18ms | 0.49ms | FP16 |
| 2048 | 0.76ms | 1.35ms | 0.73ms | cuBLAS |

**Insight**: cuBLAS INT8 only wins for large batches where the 2x compute speedup outweighs the quantization overhead.

---

## Final Implementation

### Architecture Decision

```
                    ┌─────────────────────────────────────┐
                    │           int8_gemm()               │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │         use_cublas?                 │
                    └─────────────────┬───────────────────┘
                           │                    │
                     use_cublas=True      use_cublas=False
                           │                    │
                           ▼                    ▼
              ┌────────────────────┐  ┌────────────────────┐
              │  _int8_gemm_cublas │  │  _int8_gemm_kernel │
              │  (torch._int_mm)   │  │  (Triton FP16 TC)  │
              └────────────────────┘  └────────────────────┘
              • Requires M > 16        • Works for any M
              • 1.04x speedup @ M=2048 • ~1.0x vs FP16
              • ~3-8% extra error      • Accurate
              • Uses INT8 tensor cores • Uses FP16 tensor cores
```

### Code Structure

```python
def int8_gemm(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    weight_transposed: Optional[torch.Tensor] = None,  # Pre-computed
    scale_fp16: Optional[torch.Tensor] = None,         # Pre-computed
    use_cublas: Optional[bool] = None,                 # Auto-select
) -> torch.Tensor:
    """
    W8A16 GEMM with automatic backend selection.

    Default: Triton FP16 path (accurate)
    Optional: cuBLAS INT8 path (faster for M>16, less accurate)
    """
```

### Int8Linear Module

```python
class Int8Linear(torch.nn.Module):
    """Drop-in replacement for nn.Linear with INT8 weights."""

    def __init__(self, in_features, out_features, bias=False):
        # Buffers for weights and pre-computed values
        self.weight_int8       # INT8 weights [N, K]
        self.scale             # FP32 scales [N]
        self.scale_fp16        # Pre-converted FP16 scales
        self.weight_transposed # Pre-transposed [K, N]

    def quantize_weights(self, weight):
        """Quantize FP16 weights to INT8."""
        # Handles device placement automatically

    @classmethod
    def from_linear(cls, linear):
        """Convert nn.Linear to Int8Linear."""
        # Preserves device placement
```

---

## Key Learnings

### 1. Tensor Core Selection is Critical

Triton's `tl.dot` selects tensor cores based on input dtypes:
- `float16 × float16` → FP16 tensor cores (312 TFLOPS)
- `int8 × int8` → INT8 tensor cores (624 TOPS)
- `float32 × float32` → Falls back to slow path

**Always check what tensor cores your kernel actually uses.**

### 2. W8A16 is Memory-Bound, Not Compute-Bound

For W8A16 (INT8 weights, FP16 activations):
- We can't use INT8 tensor cores directly (activations are FP16)
- Two options: dequantize weights (FP16 TC) or quantize activations (INT8 TC)
- Quantizing activations adds overhead that often cancels compute benefit
- **Main value is 2x memory reduction**, not compute speedup

### 3. cuBLAS Has Hidden Layout Requirements

`torch._int_mm` performance varies 5x based on matrix layout:
- Column-major B (transpose view): Fast path
- Row-major B (contiguous): Slow path

**Profile with actual data layouts, not synthetic benchmarks.**

### 4. Fused Kernels Don't Always Win

We tried fusing quantization/dequantization into the GEMM kernel. Result: slower than cuBLAS.

**Reasons**:
- cuBLAS is highly optimized with hand-tuned assembly
- Triton's INT8 codegen isn't as mature as FP16
- Fusion helps memory-bound ops; GEMM is compute-bound for large M

### 5. Pre-computation Amortizes Overhead

For repeated inference with the same weights:
```python
# Pre-compute once
weight_transposed = weight_int8.t().contiguous()
scale_fp16 = scale.half()

# Use pre-computed values in forward pass
y = int8_gemm(x, weight_int8, scale,
              weight_transposed=weight_transposed,
              scale_fp16=scale_fp16)
```

### 6. Accuracy vs Speed Tradeoff

| Path | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| Triton FP16 | ~1.0x | High | Default, accuracy-critical |
| cuBLAS INT8 | 1.04x | Lower (~3-8% error) | Speed-critical, large batches |

---

## Recommendations

### For Users

1. **Use default path** (`use_cublas=False`) for accuracy
2. **Use cuBLAS path** (`use_cublas=True`) only for:
   - Large batch sizes (M > 256)
   - Speed-critical applications tolerant of ~5% error
3. **Pre-compute** `weight_transposed` and `scale_fp16` for repeated inference
4. **Quantize weights once** at model load time, not per-forward

### For Future Work

1. **True W8A8**: Pre-quantize activations for INT8 tensor cores without per-call overhead
2. **SmoothQuant**: Better activation quantization with learned scales
3. **FP8**: Newer GPUs (H100) have FP8 tensor cores with less quantization error
4. **Batched GEMM**: Fuse multiple small GEMMs to amortize overhead

---

## References

- [A100 Tensor Core Performance](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- [LLM.int8() Paper](https://arxiv.org/abs/2208.07339)
- [SmoothQuant Paper](https://arxiv.org/abs/2211.10438)
- [Triton Documentation](https://triton-lang.org/)
