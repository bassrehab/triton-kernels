"""
Benchmarks for INT8 quantized matrix multiplication (W8A16 GEMM).

Compares:
1. FP16 torch.mm (baseline)
2. INT8 Triton GEMM with dequantization

Measures latency, memory bandwidth utilization, and memory savings.
"""

import torch
from typing import Optional

from benchmarks.utils import (
    get_gpu_specs,
    benchmark_fn,
    plot_roofline,
    format_results_table,
    BenchmarkResult,
)
from triton_kernels.quantized_matmul import int8_gemm
from triton_kernels.quantization import quantize_weight_per_channel


def calculate_gemm_metrics(
    M: int,
    N: int,
    K: int,
    dtype_bits: int = 16,
) -> tuple[float, float]:
    """
    Calculate FLOPs and bytes for GEMM operation.

    GEMM: C[M,N] = A[M,K] @ B[K,N]

    FLOPs: 2 * M * N * K (multiply-add)

    Bytes:
    - Read A: M * K * dtype_size
    - Read B: K * N * dtype_size
    - Write C: M * N * dtype_size

    For small M (inference), the operation is typically weight-bound,
    meaning B dominates memory traffic.

    Returns:
        (flops, bytes_accessed)
    """
    flops = 2 * M * N * K

    dtype_bytes = dtype_bits // 8
    bytes_a = M * K * dtype_bytes
    bytes_b = K * N * dtype_bytes
    bytes_c = M * N * dtype_bytes

    return flops, bytes_a + bytes_b + bytes_c


def calculate_int8_gemm_metrics(
    M: int,
    N: int,
    K: int,
) -> tuple[float, float]:
    """
    Calculate FLOPs and bytes for INT8 GEMM.

    Bytes (INT8 weights):
    - Read A (FP16): M * K * 2 bytes
    - Read B (INT8): K * N * 1 byte
    - Read scale (FP32): N * 4 bytes
    - Write C (FP16): M * N * 2 bytes

    The key savings is B being 1 byte instead of 2.
    """
    flops = 2 * M * N * K

    bytes_a = M * K * 2       # FP16 activations
    bytes_b = K * N * 1       # INT8 weights
    bytes_scale = N * 4       # FP32 scales
    bytes_c = M * N * 2       # FP16 output

    return flops, bytes_a + bytes_b + bytes_scale + bytes_c


def benchmark_fp16_gemm(
    M: int,
    N: int,
    K: int,
) -> BenchmarkResult:
    """Benchmark FP16 torch.mm (baseline)."""
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    weight = torch.randn(N, K, device="cuda", dtype=torch.float16)

    def fp16_mm(x, w):
        return torch.nn.functional.linear(x, w)

    flops, bytes_accessed = calculate_gemm_metrics(M, N, K, dtype_bits=16)

    return benchmark_fn(
        fp16_mm,
        x,
        weight,
        flops=flops,
        bytes_accessed=bytes_accessed,
        name=f"FP16 GEMM",
    )


def benchmark_int8_gemm(
    M: int,
    N: int,
    K: int,
    use_cublas: bool = False,
) -> BenchmarkResult:
    """Benchmark INT8 GEMM.

    Args:
        M, N, K: Matrix dimensions.
        use_cublas: If True, use cuBLAS INT8 tensor cores (faster but less accurate).
                    If False, use Triton FP16 dequantization (more accurate).
    """
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    weight_fp16 = torch.randn(N, K, device="cuda", dtype=torch.float16)

    # Quantize weight
    weight_int8, scale = quantize_weight_per_channel(weight_fp16.cpu())
    weight_int8 = weight_int8.cuda()
    scale = scale.cuda()

    # Pre-transpose for Triton path
    weight_t = weight_int8.t().contiguous()

    flops, bytes_accessed = calculate_int8_gemm_metrics(M, N, K)

    def int8_fn(x, w, s):
        return int8_gemm(x, w, s, weight_transposed=weight_t, use_cublas=use_cublas)

    name = f"INT8 GEMM ({'cuBLAS' if use_cublas else 'Triton'})"

    return benchmark_fn(
        int8_fn,
        x,
        weight_int8,
        scale,
        flops=flops,
        bytes_accessed=bytes_accessed,
        name=name,
    )


def run_benchmarks(
    shapes: Optional[list[tuple[int, int, int]]] = None,
    seq_lens: Optional[list[int]] = None,
    save_roofline: Optional[str] = None,
) -> list[BenchmarkResult]:
    """
    Run full INT8 GEMM benchmark suite.

    Args:
        shapes: List of (M, N, K) tuples to benchmark.
        seq_lens: Sequence lengths for LLaMA-style benchmarks.
        save_roofline: Path to save roofline plot.

    Returns:
        List of benchmark results.
    """
    gpu_specs = get_gpu_specs()
    print(f"\n{'='*60}")
    print(f"INT8 GEMM Benchmark Suite")
    print(f"{'='*60}")
    print(gpu_specs)
    print(f"{'='*60}\n")

    all_results = []

    # Test typical LLM weight shapes with different sequence lengths
    if seq_lens is None:
        seq_lens = [1, 32, 128, 512, 2048]

    llm_shapes = [
        # (description, N, K) - M comes from seq_len
        ("Hidden→Hidden (4096x4096)", 4096, 4096),
        ("Hidden→FFN (4096→11008)", 11008, 4096),
        ("FFN→Hidden (11008→4096)", 4096, 11008),
        ("Hidden→Vocab (4096→32000)", 32000, 4096),
    ]

    print("Testing LLM-style weight shapes with varying sequence lengths:\n")

    for desc, N, K in llm_shapes:
        print(f"\n{'='*60}")
        print(f"{desc}")
        print(f"Weight shape: ({N}, {K})")
        print(f"{'='*60}")

        # Memory footprint comparison
        fp16_bytes = N * K * 2
        int8_bytes = N * K * 1 + N * 4  # weights + scales
        savings = (fp16_bytes - int8_bytes) / fp16_bytes * 100
        print(f"Memory: FP16={fp16_bytes/1e6:.1f}MB, INT8={int8_bytes/1e6:.1f}MB ({savings:.1f}% reduction)")

        for seq_len in seq_lens:
            M = seq_len
            print(f"\n--- SeqLen={M} ---")

            results = [
                benchmark_fp16_gemm(M, N, K),
                benchmark_int8_gemm(M, N, K),
            ]

            for r in results:
                print(r)

            # Calculate speedup
            fp16_ms = results[0].mean_ms
            int8_ms = results[1].mean_ms
            speedup = fp16_ms / int8_ms

            print(f"Speedup (INT8 vs FP16): {speedup:.2f}x")

            all_results.extend(results)

    # Custom shapes if provided
    if shapes:
        print(f"\n{'='*60}")
        print("Custom Shapes")
        print(f"{'='*60}")

        for M, N, K in shapes:
            print(f"\n--- M={M}, N={N}, K={K} ---")

            results = [
                benchmark_fp16_gemm(M, N, K),
                benchmark_int8_gemm(M, N, K),
            ]

            for r in results:
                print(r)

            all_results.extend(results)

    # Summary
    print(f"\n{'='*60}")
    print("Summary Table")
    print(f"{'='*60}")
    print(format_results_table(all_results, baseline_name="FP16"))

    # Analysis
    print(f"\n{'='*60}")
    print("Analysis: When INT8 GEMM Wins")
    print(f"{'='*60}")
    print("""
INT8 GEMM is most effective when:

1. **Memory-bound (low seq_len)**: For seq_len=1-32, the operation is
   weight-bound. INT8 weights = 2x less memory traffic = ~1.5-2x speedup.

2. **Large weight matrices**: The larger the weight (e.g., vocab projection),
   the more memory traffic savings matter.

3. **Not compute-bound**: For very long sequences (seq_len > 1024), the
   operation becomes compute-bound and INT8 advantage diminishes.

The sweet spot is inference with batch_size=1 and seq_len < 256, which is
the typical autoregressive decoding scenario.
""")

    # Generate roofline plot
    if save_roofline:
        plot_roofline(
            all_results,
            gpu_specs=gpu_specs,
            title="INT8 GEMM Roofline Analysis",
            save_path=save_roofline,
            show=False,
        )

    return all_results


def main():
    """Run benchmarks with default settings."""
    import argparse

    parser = argparse.ArgumentParser(description="INT8 GEMM Benchmark Suite")
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[1, 32, 128, 512, 2048],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--save-roofline",
        type=str,
        default=None,
        help="Path to save roofline plot",
    )
    args = parser.parse_args()

    run_benchmarks(
        seq_lens=args.seq_lens,
        save_roofline=args.save_roofline,
    )


if __name__ == "__main__":
    main()
