"""
Benchmarks for RMSNorm kernels.

Compares:
1. PyTorch RMSNorm (baseline)
2. Triton RMSNorm
3. Triton fused RMSNorm + residual

Measures latency, bandwidth utilization, and generates roofline analysis.
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
from triton_kernels.rmsnorm import (
    rmsnorm,
    rmsnorm_torch,
    rmsnorm_residual_fused,
)


def calculate_rmsnorm_metrics(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.float16,
) -> tuple[float, float]:
    """
    Calculate FLOPs and bytes for RMSNorm operation.

    RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight

    FLOPs per element:
    - x^2: 1 FLOP
    - sum: 1 FLOP (reduction)
    - mean (div by N): 1 FLOP (amortized)
    - add eps: 1 FLOP (amortized)
    - rsqrt: 1 FLOP (amortized)
    - x * rrms: 1 FLOP
    - * weight: 1 FLOP
    Total: ~4 FLOPs per element

    Bytes:
    - Read x: N * dtype_size
    - Read weight: N * dtype_size
    - Write y: N * dtype_size
    Total: 3N * dtype_size

    Returns:
        (flops, bytes_accessed)
    """
    num_elements = batch_size * seq_len * hidden_dim
    dtype_size = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    flops = 4 * num_elements
    bytes_accessed = 3 * num_elements * dtype_size

    return flops, bytes_accessed


def calculate_rmsnorm_residual_metrics(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.float16,
) -> tuple[float, float]:
    """
    Calculate FLOPs and bytes for fused RMSNorm + residual.

    Fused: y = RMSNorm(x + residual) * weight

    FLOPs per element:
    - x + residual: 1 FLOP
    - (x+r)^2: 1 FLOP
    - sum, mean, rsqrt: amortized
    - (x+r) * rrms: 1 FLOP
    - * weight: 1 FLOP
    Total: ~5 FLOPs per element

    Bytes (fused):
    - Read x: N * dtype_size
    - Read residual: N * dtype_size
    - Read weight: N * dtype_size
    - Write y: N * dtype_size
    Total: 4N * dtype_size

    Returns:
        (flops, bytes_accessed)
    """
    num_elements = batch_size * seq_len * hidden_dim
    dtype_size = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    flops = 5 * num_elements
    bytes_accessed = 4 * num_elements * dtype_size

    return flops, bytes_accessed


def benchmark_pytorch_rmsnorm(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.float16,
) -> BenchmarkResult:
    """Benchmark PyTorch reference RMSNorm."""
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_dim, device="cuda", dtype=dtype)

    flops, bytes_accessed = calculate_rmsnorm_metrics(batch_size, seq_len, hidden_dim, dtype)

    return benchmark_fn(
        rmsnorm_torch,
        x,
        weight,
        1e-6,
        flops=flops,
        bytes_accessed=bytes_accessed,
        name="RMSNorm (PyTorch)",
    )


def benchmark_triton_rmsnorm(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.float16,
) -> BenchmarkResult:
    """Benchmark Triton RMSNorm kernel."""
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_dim, device="cuda", dtype=dtype)

    flops, bytes_accessed = calculate_rmsnorm_metrics(batch_size, seq_len, hidden_dim, dtype)

    return benchmark_fn(
        rmsnorm,
        x,
        weight,
        1e-6,
        flops=flops,
        bytes_accessed=bytes_accessed,
        name="RMSNorm (Triton)",
    )


def benchmark_pytorch_rmsnorm_residual(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.float16,
) -> BenchmarkResult:
    """Benchmark PyTorch separate add + RMSNorm (baseline for fusion)."""
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_dim, device="cuda", dtype=dtype)

    def pytorch_separate(x, residual, weight, eps):
        return rmsnorm_torch(x + residual, weight, eps)

    # Separate ops have extra memory traffic for intermediate
    num_elements = batch_size * seq_len * hidden_dim
    dtype_size = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    # Separate: write x+r to memory, then read it back
    flops = 5 * num_elements
    bytes_accessed = 5 * num_elements * dtype_size  # Extra read/write for intermediate

    return benchmark_fn(
        pytorch_separate,
        x,
        residual,
        weight,
        1e-6,
        flops=flops,
        bytes_accessed=bytes_accessed,
        name="RMSNorm+Residual (PyTorch)",
    )


def benchmark_triton_rmsnorm_residual_fused(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.float16,
) -> BenchmarkResult:
    """Benchmark Triton fused RMSNorm + residual kernel."""
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_dim, device="cuda", dtype=dtype)

    flops, bytes_accessed = calculate_rmsnorm_residual_metrics(
        batch_size, seq_len, hidden_dim, dtype
    )

    return benchmark_fn(
        rmsnorm_residual_fused,
        x,
        residual,
        weight,
        1e-6,
        flops=flops,
        bytes_accessed=bytes_accessed,
        name="RMSNorm+Residual (Triton Fused)",
    )


def run_benchmarks(
    hidden_dims: Optional[list[int]] = None,
    seq_lens: Optional[list[int]] = None,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float16,
    save_roofline: Optional[str] = None,
) -> list[BenchmarkResult]:
    """
    Run full RMSNorm benchmark suite.

    Args:
        hidden_dims: Hidden dimensions to test.
        seq_lens: Sequence lengths to test.
        batch_size: Batch size (default 1 for inference).
        dtype: Data type.
        save_roofline: Path to save roofline plot.

    Returns:
        List of benchmark results.
    """
    if hidden_dims is None:
        hidden_dims = [1024, 2048, 4096, 8192]
    if seq_lens is None:
        seq_lens = [512, 1024, 2048, 4096]

    gpu_specs = get_gpu_specs()
    print(f"\n{'='*60}")
    print(f"RMSNorm Benchmark Suite")
    print(f"{'='*60}")
    print(gpu_specs)
    print(f"\nBatch size: {batch_size}")
    print(f"Dtype: {dtype}")
    print(f"{'='*60}\n")

    all_results = []

    for hidden_dim in hidden_dims:
        for seq_len in seq_lens:
            print(f"\n--- Hidden={hidden_dim}, SeqLen={seq_len} ---")

            # Run benchmarks
            results = [
                benchmark_pytorch_rmsnorm(batch_size, seq_len, hidden_dim, dtype),
                benchmark_triton_rmsnorm(batch_size, seq_len, hidden_dim, dtype),
                benchmark_pytorch_rmsnorm_residual(batch_size, seq_len, hidden_dim, dtype),
                benchmark_triton_rmsnorm_residual_fused(batch_size, seq_len, hidden_dim, dtype),
            ]

            # Print results
            for r in results:
                print(r)

            # Calculate speedups
            pytorch_ms = results[0].mean_ms
            triton_ms = results[1].mean_ms
            pytorch_fused_ms = results[2].mean_ms
            triton_fused_ms = results[3].mean_ms

            print(f"\nSpeedups:")
            print(f"  Triton vs PyTorch RMSNorm: {pytorch_ms/triton_ms:.2f}x")
            print(f"  Triton Fused vs PyTorch Separate: {pytorch_fused_ms/triton_fused_ms:.2f}x")

            all_results.extend(results)

    # Generate summary table
    print(f"\n{'='*60}")
    print("Summary Table")
    print(f"{'='*60}")
    print(format_results_table(all_results, baseline_name="PyTorch"))

    # Generate roofline plot
    if save_roofline:
        plot_roofline(
            all_results,
            gpu_specs=gpu_specs,
            title="RMSNorm Roofline Analysis",
            save_path=save_roofline,
            show=False,
        )

    return all_results


def main():
    """Run benchmarks with default settings."""
    import argparse

    parser = argparse.ArgumentParser(description="RMSNorm Benchmark Suite")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
        help="Hidden dimensions to benchmark",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1 for inference)",
    )
    parser.add_argument(
        "--save-roofline",
        type=str,
        default=None,
        help="Path to save roofline plot",
    )
    args = parser.parse_args()

    run_benchmarks(
        hidden_dims=args.hidden_dims,
        seq_lens=args.seq_lens,
        batch_size=args.batch_size,
        save_roofline=args.save_roofline,
    )


if __name__ == "__main__":
    main()
