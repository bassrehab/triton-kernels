# INT8 GEMM Performance Investigation

Notes from debugging why the INT8 kernel was slower than FP16 on A100.

## The Problem

Ran benchmarks and got this:

```
FP16 GEMM: 0.76ms, 242 TFLOPS
INT8 GEMM: 2.43ms,  76 TFLOPS  <- wtf, 3x SLOWER
```

INT8 tensor cores should be 2x faster (624 TOPS vs 312 TFLOPS). Something was very wrong.

| Metric | Expected | Actual |
|--------|----------|--------|
| INT8 vs FP16 speedup | 1.5-2x | 0.31x (3x slower) |
| INT8 TFLOPS | ~400-500 | 76 |

## Root Cause

After way too much printf debugging, found three bugs:

**Bug 1: Wrong tensor cores**

```python
# This was the problem - converts to FP32 which doesn't use tensor cores!
b_fp32 = b_int8.to(tl.float32)
acc += tl.dot(a.to(tl.float32), b_fp32)
```

A100 tensor cores only do: FP16xFP16->FP32, BF16xBF16->FP32, INT8xINT8->INT32. No FP32 matmul on tensor cores.

**Bug 2: Fixed block sizes** - wasn't autotuning, just hardcoded 64x64x32.

**Bug 3: Transpose every call** - `weight_int8.t().contiguous()` was in the forward pass, creating a copy every time.

## What I Tried

### Approach 1: Dequantize to FP16, use FP16 tensor cores

```python
b_int8 = tl.load(b_ptrs, mask=b_mask, other=0)
b_fp16 = b_int8.to(tl.float16)  # dequant in registers
acc += tl.dot(a, b_fp16, out_dtype=tl.float32)
```

Result: matches FP16 perf (~0.76ms). No speedup but at least not slower anymore. The INT8->FP16 conversion is basically free since it happens in registers.

### Approach 2: cuBLAS INT8 via torch._int_mm

Quantize activations on the fly, use actual INT8 tensor cores:

```python
x_int8 = (x * scale).round().clamp(-128, 127).to(torch.int8)
y_int32 = torch._int_mm(x_int8, weight_int8.t())
y_fp16 = (y_int32.float() * combined_scale).half()
```

Results:
- Just the matmul: 0.45ms (413 TOPS, 66% peak)
- Full pipeline with quant/dequant: 0.73ms (1.04x vs FP16)

Catches: adds 3-8% numerical error from quantizing activations.

### Approach 3: Fused Triton kernel

Tried fusing the activation quant into the kernel. Got 1.2ms - slower than just calling cuBLAS. Triton's INT8 codegen isn't as optimized.

## Profiling Breakdown

For M=2048, K=4096, N=11008:

```
absmax computation      0.053ms     5%
Activation quantization 0.133ms    12%
INT8 matmul (_int_mm)   0.447ms    42%
Dequantization          0.367ms    34%
Other                   0.097ms     9%
-----------------------------------------
Total INT8 pipeline     1.097ms
FP16 baseline           0.912ms
```

The matmul itself IS 2x faster. But quant/dequant overhead (0.55ms) eats all the gains.

## Gotcha: cuBLAS layout sensitivity

Spent an hour on this one. `torch._int_mm` is 5x slower if you pass the wrong layout:

```python
# SLOW (2.7ms) - B is row-major
y = torch._int_mm(A, B_contiguous)

# FAST (0.5ms) - B is column-major (transpose VIEW, not contiguous!)
y = torch._int_mm(A, B.t())
```

Don't call `.contiguous()` on the transpose.

## Batch size matters

| Batch (M) | FP16 | INT8 Triton | INT8 cuBLAS | Best |
|-----------|------|-------------|-------------|------|
| 1 | 0.09ms | 0.09ms | N/A | tie |
| 32 | 0.09ms | 0.09ms | 0.51ms | Triton |
| 128 | 0.11ms | 0.18ms | 0.49ms | FP16 |
| 2048 | 0.76ms | 1.35ms | 0.73ms | cuBLAS |

cuBLAS INT8 only wins for large batches. For single token inference, doesn't matter.

## Final Design

Ended up with two paths:
- Default (`use_cublas=False`): Triton kernel, dequant weights to FP16, accurate
- Optional (`use_cublas=True`): cuBLAS INT8, faster for M>16, less accurate

Pre-compute transposed weights and fp16 scales to avoid per-call overhead.

## Takeaways

1. **Check your tensor cores** - tl.dot dtype selection is subtle. FP32 inputs = no tensor cores.

2. **W8A16 is memory-bound** - can't use INT8 tensor cores without quantizing activations. Main benefit is 2x memory reduction, not compute speedup.

3. **cuBLAS layout matters** - column-major B is fast path. Transpose view (not contiguous) triggers it.

4. **Fusion doesn't always win** - cuBLAS has hand-tuned assembly. Triton INT8 isn't there yet.

5. **Profile the whole pipeline** - the matmul being 2x faster means nothing if overhead cancels it.

## Future ideas

- W8A8 with pre-quantized activations
- SmoothQuant for better activation distribution
- FP8 on H100

## References

- https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
- https://arxiv.org/abs/2208.07339 (LLM.int8)
- https://arxiv.org/abs/2211.10438 (SmoothQuant)
