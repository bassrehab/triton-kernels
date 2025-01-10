"""
Benchmark suite for triton-kernels.

Run individual benchmarks:
    python -m benchmarks.bench_rmsnorm
    python -m benchmarks.bench_swiglu
    python -m benchmarks.bench_quantized_matmul

Generate roofline analysis:
    python -m benchmarks.roofline
"""

from benchmarks.utils import (
    GPUSpecs,
    BenchmarkResult,
    get_gpu_specs,
    benchmark_fn,
    calculate_arithmetic_intensity,
    plot_roofline,
    format_results_table,
    print_gpu_info,
)

__all__ = [
    "GPUSpecs",
    "BenchmarkResult",
    "get_gpu_specs",
    "benchmark_fn",
    "calculate_arithmetic_intensity",
    "plot_roofline",
    "format_results_table",
    "print_gpu_info",
]
