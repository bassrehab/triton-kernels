"""
Benchmarks for SwiGLU activation kernels.

Compares:
1. PyTorch baseline (separate F.silu + multiply)
2. Triton fused SwiGLU

Measures latency, bandwidth utilization, and generates roofline analysis.
"""

import torch
import torch.nn.functional as F
from typing import Optional

from benchmarks.utils import (
    get_gpu_specs,
    benchmark_fn,
    plot_roofline,
    format_results_table,
    BenchmarkResult,
)
from triton_kernels.swiglu import swiglu_fused, swiglu_torch


def calculate_swiglu_metrics(
    batch_size: int,
    seq_len: int,
    ffn_dim: int,
    dtype: torch.dtype = torch.float16,
    fused: bool = True,
) -> tuple[float, float]:
    """
    Calculate FLOPs and bytes for SwiGLU operation.

    SwiGLU: y = silu(gate) * up = gate * sigmoid(gate) * up

    FLOPs per element:
    - sigmoid(gate): ~4 ops (exp, neg, add 1, div)
    - gate * sigmoid: 1 mul
    - * up: 1 mul
    Total: ~6 FLOPs per element

    Bytes (fused):
    - Read gate: N * dtype_size
    - Read up: N * dtype_size
    - Write y: N * dtype_size
    Total: 3N * dtype_size

    Bytes (separate):
    - Read gate: N * dtype_size
    - Write silu_gate: N * dtype_size (intermediate)
    - Read silu_gate: N * dtype_size
    - Read up: N * dtype_size
    - Write y: N * dtype_size
    Total: 5N * dtype_size

    Returns:
        (flops, bytes_accessed)
    """
    num_elements = batch_size * seq_len * ffn_dim
    dtype_size = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    # FLOPs include sigmoid computation
    flops = 6 * num_elements

    if fused:
        # Fused: read gate, read up, write output
        bytes_accessed = 3 * num_elements * dtype_size
    else:
        # Separate: extra read/write for intermediate silu(gate)
        bytes_accessed = 5 * num_elements * dtype_size

    return flops, bytes_accessed


def benchmark_pytorch_swiglu(
    batch_size: int,
    seq_len: int,
    ffn_dim: int,
    dtype: torch.dtype = torch.float16,
) -> BenchmarkResult:
    """Benchmark PyTorch separate silu + multiply."""
    gate = torch.randn(batch_size, seq_len, ffn_dim, device="cuda", dtype=dtype)
    up = torch.randn_like(gate)

    def pytorch_swiglu(gate, up):
        return F.silu(gate) * up

    flops, bytes_accessed = calculate_swiglu_metrics(
        batch_size, seq_len, ffn_dim, dtype, fused=False
    )

    return benchmark_fn(
        pytorch_swiglu,
        gate,
        up,
        flops=flops,
        bytes_accessed=bytes_accessed,
        name="SwiGLU (PyTorch)",
    )


def benchmark_triton_swiglu(
    batch_size: int,
    seq_len: int,
    ffn_dim: int,
    dtype: torch.dtype = torch.float16,
) -> BenchmarkResult:
    """Benchmark Triton fused SwiGLU kernel."""
    gate = torch.randn(batch_size, seq_len, ffn_dim, device="cuda", dtype=dtype)
    up = torch.randn_like(gate)

    flops, bytes_accessed = calculate_swiglu_metrics(
        batch_size, seq_len, ffn_dim, dtype, fused=True
    )

    return benchmark_fn(
        swiglu_fused,
        gate,
        up,
        flops=flops,
        bytes_accessed=bytes_accessed,
        name="SwiGLU (Triton Fused)",
    )


def run_benchmarks(
    configs: Optional[list[tuple[int, int]]] = None,
    batch_size: int = 1,
    seq_len: int = 2048,
    dtype: torch.dtype = torch.float16,
    save_roofline: Optional[str] = None,
) -> list[BenchmarkResult]:
    """
    Run full SwiGLU benchmark suite.

    Args:
        configs: List of (hidden_dim, ffn_dim) tuples. Defaults to LLaMA configs.
        batch_size: Batch size (default 1 for inference).
        seq_len: Sequence length.
        dtype: Data type.
        save_roofline: Path to save roofline plot.

    Returns:
        List of benchmark results.
    """
    if configs is None:
        # LLaMA-style configurations: (hidden_dim, ffn_dim)
        # Note: We only use ffn_dim for the activation, hidden_dim is for context
        configs = [
            (4096, 11008),    # LLaMA 7B
            (5120, 13824),    # LLaMA 13B
            (6656, 17920),    # LLaMA 33B (approx)
            (8192, 22016),    # LLaMA 65B (approx)
        ]

    gpu_specs = get_gpu_specs()
    print(f"\n{'='*60}")
    print(f"SwiGLU Benchmark Suite")
    print(f"{'='*60}")
    print(gpu_specs)
    print(f"\nBatch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Dtype: {dtype}")
    print(f"{'='*60}\n")

    all_results = []

    for hidden_dim, ffn_dim in configs:
        print(f"\n--- Hidden={hidden_dim}, FFN={ffn_dim} (LLaMA-style) ---")

        # Run benchmarks
        results = [
            benchmark_pytorch_swiglu(batch_size, seq_len, ffn_dim, dtype),
            benchmark_triton_swiglu(batch_size, seq_len, ffn_dim, dtype),
        ]

        # Print results
        for r in results:
            print(r)

        # Calculate speedup
        pytorch_ms = results[0].mean_ms
        triton_ms = results[1].mean_ms

        print(f"\nSpeedup (Triton vs PyTorch): {pytorch_ms/triton_ms:.2f}x")

        # Memory savings
        pytorch_bytes = calculate_swiglu_metrics(batch_size, seq_len, ffn_dim, dtype, fused=False)[1]
        triton_bytes = calculate_swiglu_metrics(batch_size, seq_len, ffn_dim, dtype, fused=True)[1]
        memory_savings = (pytorch_bytes - triton_bytes) / pytorch_bytes * 100
        print(f"Memory traffic reduction: {memory_savings:.1f}%")

        all_results.extend(results)

    # Also test varying sequence lengths
    print(f"\n{'='*60}")
    print("Varying Sequence Length (FFN=11008, LLaMA 7B)")
    print(f"{'='*60}")

    ffn_dim = 11008
    for test_seq_len in [512, 1024, 2048, 4096]:
        print(f"\n--- SeqLen={test_seq_len} ---")

        results = [
            benchmark_pytorch_swiglu(batch_size, test_seq_len, ffn_dim, dtype),
            benchmark_triton_swiglu(batch_size, test_seq_len, ffn_dim, dtype),
        ]

        for r in results:
            print(r)

        pytorch_ms = results[0].mean_ms
        triton_ms = results[1].mean_ms
        print(f"Speedup: {pytorch_ms/triton_ms:.2f}x")

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
            title="SwiGLU Roofline Analysis",
            save_path=save_roofline,
            show=False,
        )

    return all_results


def main():
    """Run benchmarks with default settings."""
    import argparse

    parser = argparse.ArgumentParser(description="SwiGLU Benchmark Suite")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1 for inference)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)",
    )
    parser.add_argument(
        "--save-roofline",
        type=str,
        default=None,
        help="Path to save roofline plot",
    )
    args = parser.parse_args()

    run_benchmarks(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        save_roofline=args.save_roofline,
    )


if __name__ == "__main__":
    main()
