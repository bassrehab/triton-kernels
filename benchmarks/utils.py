"""
Benchmarking utilities for triton-kernels.

Provides:
- GPU detection and spec reporting
- Roofline model plotting
- Benchmark runner with warmup
- Arithmetic intensity calculation
"""

from dataclasses import dataclass
from typing import Callable, Optional
import torch
import triton


# Known GPU specifications (memory bandwidth in GB/s, peak TFLOPS for FP16)
# These are theoretical peaks - actual achievable is typically 70-85%
# TODO: fetch specs dynamically from nvidia-smi or cuda APIs
GPU_SPECS = {
    # NVIDIA Data Center
    "A100-SXM4-40GB": {"bandwidth_gb_s": 1555, "fp16_tflops": 312},
    "A100-SXM4-80GB": {"bandwidth_gb_s": 2039, "fp16_tflops": 312},
    "A100-PCIE-40GB": {"bandwidth_gb_s": 1555, "fp16_tflops": 312},
    "A100-PCIE-80GB": {"bandwidth_gb_s": 1935, "fp16_tflops": 312},
    "H100-SXM5": {"bandwidth_gb_s": 3350, "fp16_tflops": 989},
    "H100-PCIE": {"bandwidth_gb_s": 2000, "fp16_tflops": 756},
    "V100-SXM2-32GB": {"bandwidth_gb_s": 900, "fp16_tflops": 125},
    "V100-PCIE-32GB": {"bandwidth_gb_s": 900, "fp16_tflops": 125},

    # NVIDIA Consumer
    "NVIDIA GeForce RTX 4090": {"bandwidth_gb_s": 1008, "fp16_tflops": 165},
    "NVIDIA GeForce RTX 4080": {"bandwidth_gb_s": 717, "fp16_tflops": 97},
    "NVIDIA GeForce RTX 3090": {"bandwidth_gb_s": 936, "fp16_tflops": 71},
    "NVIDIA GeForce RTX 3090 Ti": {"bandwidth_gb_s": 1008, "fp16_tflops": 80},
    "NVIDIA GeForce RTX 3080": {"bandwidth_gb_s": 760, "fp16_tflops": 60},
    "NVIDIA GeForce RTX 3080 Ti": {"bandwidth_gb_s": 912, "fp16_tflops": 68},

    # Fallback defaults
    "default": {"bandwidth_gb_s": 900, "fp16_tflops": 100},
}


@dataclass
class GPUSpecs:
    """GPU hardware specifications."""
    name: str
    compute_capability: tuple[int, int]
    total_memory_gb: float
    memory_bandwidth_gb_s: float
    fp16_tflops: float
    num_sms: int

    def __str__(self) -> str:
        return (
            f"GPU: {self.name}\n"
            f"  Compute Capability: {self.compute_capability[0]}.{self.compute_capability[1]}\n"
            f"  Memory: {self.total_memory_gb:.1f} GB\n"
            f"  Memory Bandwidth: {self.memory_bandwidth_gb_s:.0f} GB/s\n"
            f"  FP16 Peak: {self.fp16_tflops:.0f} TFLOPS\n"
            f"  SMs: {self.num_sms}"
        )


def get_gpu_specs() -> GPUSpecs:
    """
    Detect GPU and return hardware specifications.

    Returns:
        GPUSpecs object with detected GPU information.

    Raises:
        RuntimeError: If no CUDA-capable GPU is detected.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA-capable GPU detected")

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    name = props.name
    compute_capability = (props.major, props.minor)
    total_memory_gb = props.total_memory / (1024 ** 3)
    num_sms = props.multi_processor_count

    # Look up known specs or use defaults
    specs = GPU_SPECS.get(name, None)
    if specs is None:
        # Try partial match
        for known_name, known_specs in GPU_SPECS.items():
            if known_name in name or name in known_name:
                specs = known_specs
                break

    if specs is None:
        specs = GPU_SPECS["default"]
        print(f"Warning: Unknown GPU '{name}', using default specs")

    return GPUSpecs(
        name=name,
        compute_capability=compute_capability,
        total_memory_gb=total_memory_gb,
        memory_bandwidth_gb_s=specs["bandwidth_gb_s"],
        fp16_tflops=specs["fp16_tflops"],
        num_sms=num_sms,
    )


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    flops: Optional[float] = None
    bytes_accessed: Optional[float] = None

    @property
    def arithmetic_intensity(self) -> Optional[float]:
        """FLOPs per byte of memory traffic."""
        if self.flops is not None and self.bytes_accessed is not None and self.bytes_accessed > 0:
            return self.flops / self.bytes_accessed
        return None

    @property
    def achieved_bandwidth_gb_s(self) -> Optional[float]:
        """Achieved memory bandwidth in GB/s."""
        if self.bytes_accessed is not None and self.mean_ms > 0:
            return (self.bytes_accessed / 1e9) / (self.mean_ms / 1000)
        return None

    @property
    def achieved_tflops(self) -> Optional[float]:
        """Achieved compute throughput in TFLOPS."""
        if self.flops is not None and self.mean_ms > 0:
            return (self.flops / 1e12) / (self.mean_ms / 1000)
        return None

    def __str__(self) -> str:
        s = f"{self.name}: {self.mean_ms:.3f} ± {self.std_ms:.3f} ms"
        if self.achieved_bandwidth_gb_s is not None:
            s += f" | {self.achieved_bandwidth_gb_s:.1f} GB/s"
        if self.achieved_tflops is not None:
            s += f" | {self.achieved_tflops:.2f} TFLOPS"
        if self.arithmetic_intensity is not None:
            s += f" | AI={self.arithmetic_intensity:.2f}"
        return s


def benchmark_fn(
    fn: Callable,
    *args,
    warmup: int = 25,
    rep: int = 100,
    quantiles: Optional[list[float]] = None,
    flops: Optional[float] = None,
    bytes_accessed: Optional[float] = None,
    name: str = "kernel",
    **kwargs,
) -> BenchmarkResult:
    """
    Benchmark a function with GPU warmup.

    Uses triton.testing.do_bench for accurate GPU timing.

    Args:
        fn: Function to benchmark (should be a CUDA kernel or operation).
        *args: Positional arguments to pass to fn.
        warmup: Number of warmup iterations.
        rep: Number of timed iterations.
        quantiles: Quantiles to compute (default: [0.5, 0.2, 0.8]).
        flops: Total FLOPs for arithmetic intensity calculation.
        bytes_accessed: Total bytes read/written for bandwidth calculation.
        name: Name for this benchmark (for display).
        **kwargs: Keyword arguments to pass to fn.

    Returns:
        BenchmarkResult with timing statistics.
    """
    if quantiles is None:
        quantiles = [0.5, 0.2, 0.8]

    # Create a wrapper that calls fn with args/kwargs
    def bench_fn():
        return fn(*args, **kwargs)

    # Run benchmark using Triton's do_bench
    # subhadipmitra, 2026-03-21: Triton 3.0 do_bench doesn't support return_mode="all".
    # Use quantiles to get median, 20th, 80th percentile instead
    try:
        ms, min_ms, max_ms = triton.testing.do_bench(
            bench_fn, warmup=warmup, rep=rep,
            quantiles=quantiles, return_mode="all",
        )
    except (AssertionError, TypeError):
        # Triton 3.0 path: use quantiles parameter
        result = triton.testing.do_bench(
            bench_fn, warmup=warmup, rep=rep,
            quantiles=quantiles,
        )
        if isinstance(result, (list, tuple)) and len(result) == 3:
            ms, min_ms, max_ms = result
        else:
            ms = float(result)
            min_ms = ms
            max_ms = ms

    # Estimate std from quantiles (rough approximation)
    std_ms = (max_ms - min_ms) / 2.0

    return BenchmarkResult(
        name=name,
        mean_ms=ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        flops=flops,
        bytes_accessed=bytes_accessed,
    )


def calculate_arithmetic_intensity(flops: float, bytes_accessed: float) -> float:
    """
    Calculate arithmetic intensity (FLOPs per byte).

    Args:
        flops: Total floating-point operations.
        bytes_accessed: Total bytes read and written.

    Returns:
        Arithmetic intensity (FLOPs/byte).

    Example:
        For RMSNorm with hidden_dim=4096:
        - Read x: 4096 * 2 bytes (FP16)
        - Read weight: 4096 * 2 bytes
        - Write y: 4096 * 2 bytes
        - FLOPs: ~6 * 4096 (square, sum, rsqrt, mul, mul)
        AI = (6 * 4096) / (3 * 4096 * 2) ≈ 1.0
    """
    if bytes_accessed == 0:
        return float('inf')
    return flops / bytes_accessed


def plot_roofline(
    results: list[BenchmarkResult],
    gpu_specs: Optional[GPUSpecs] = None,
    title: str = "Roofline Analysis",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple[float, float] = (10, 7),
) -> None:
    """
    Generate a publication-quality roofline plot.

    The roofline model shows the relationship between arithmetic intensity
    and achievable performance, bounded by memory bandwidth and compute limits.

    Args:
        results: List of BenchmarkResult objects to plot.
        gpu_specs: GPU specifications (auto-detected if None).
        title: Plot title.
        save_path: Path to save figure (PNG, PDF, etc.).
        show: Whether to display the plot.
        figsize: Figure size in inches.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib and numpy required for plotting. Install with: pip install matplotlib numpy")

    if gpu_specs is None:
        gpu_specs = get_gpu_specs()

    # Set up publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Roofline parameters
    peak_bandwidth = gpu_specs.memory_bandwidth_gb_s  # GB/s
    peak_compute = gpu_specs.fp16_tflops * 1000  # GFLOPS

    # Ridge point: where memory and compute rooflines meet
    ridge_point = peak_compute / peak_bandwidth  # FLOPs/byte

    # Create roofline
    ai_range = np.logspace(-2, 4, 1000)

    # Memory-bound region: performance = bandwidth * AI
    memory_bound = peak_bandwidth * ai_range

    # Compute-bound region: performance = peak_compute
    compute_bound = np.full_like(ai_range, peak_compute)

    # Actual roofline is the minimum
    roofline = np.minimum(memory_bound, compute_bound)

    # Plot roofline
    ax.loglog(ai_range, roofline, 'k-', linewidth=2.5, label='Roofline', zorder=1)

    # Fill regions
    ax.fill_between(ai_range, roofline, 1e-1, alpha=0.1, color='gray')

    # Add ridge point annotation
    ax.axvline(x=ridge_point, color='gray', linestyle='--', alpha=0.5)
    ax.annotate(f'Ridge Point\nAI={ridge_point:.1f}',
                xy=(ridge_point, peak_compute),
                xytext=(ridge_point * 2, peak_compute * 0.5),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

    # Color palette for different kernel types
    colors = {
        'rmsnorm': '#2ecc71',      # Green
        'swiglu': '#3498db',        # Blue
        'quantized': '#e74c3c',     # Red
        'attention': '#9b59b6',     # Purple
        'baseline': '#95a5a6',      # Gray
        'default': '#f39c12',       # Orange
    }

    markers = {
        'rmsnorm': 'o',
        'swiglu': 's',
        'quantized': '^',
        'attention': 'D',
        'baseline': 'x',
        'default': 'o',
    }

    # Plot each result
    plotted_categories = set()
    for result in results:
        if result.arithmetic_intensity is None or result.achieved_tflops is None:
            continue

        ai = result.arithmetic_intensity
        gflops = result.achieved_tflops * 1000  # Convert TFLOPS to GFLOPS

        # Determine category from name
        category = 'default'
        name_lower = result.name.lower()
        for cat in colors.keys():
            if cat in name_lower:
                category = cat
                break

        color = colors[category]
        marker = markers[category]

        ax.scatter(ai, gflops, c=color, marker=marker, s=120,
                   edgecolors='black', linewidths=0.5, zorder=3)

        # Calculate efficiency
        theoretical_max = min(peak_bandwidth * ai, peak_compute)
        efficiency = (gflops / theoretical_max) * 100

        # Annotate point
        label = f"{result.name}\n({efficiency:.0f}% eff)"
        ax.annotate(label, (ai, gflops),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, ha='left')

        plotted_categories.add(category)

    # Create legend for categories
    legend_handles = []
    for cat in plotted_categories:
        if cat != 'default':
            handle = mpatches.Patch(color=colors[cat], label=cat.replace('_', ' ').title())
            legend_handles.append(handle)

    if legend_handles:
        ax.legend(handles=legend_handles, loc='lower right', fontsize=9)

    # Labels and title
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title(f'{title}\n{gpu_specs.name}', fontsize=14, fontweight='bold')

    # Set axis limits
    ax.set_xlim(1e-1, 1e3)
    ax.set_ylim(1e1, peak_compute * 2)

    # Add hardware specs annotation
    specs_text = (
        f"Peak BW: {peak_bandwidth:.0f} GB/s\n"
        f"Peak FP16: {peak_compute/1000:.0f} TFLOPS"
    )
    ax.text(0.02, 0.98, specs_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Region labels
    ax.text(0.3, peak_compute * 0.3, 'Memory\nBound', fontsize=10,
            ha='center', va='center', alpha=0.7, style='italic')
    ax.text(ridge_point * 10, peak_compute * 0.7, 'Compute\nBound', fontsize=10,
            ha='center', va='center', alpha=0.7, style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved roofline plot to: {save_path}")

    if show:
        plt.show()

    plt.close()


def format_results_table(
    results: list[BenchmarkResult],
    baseline_name: Optional[str] = None,
) -> str:
    """
    Format benchmark results as a markdown table.

    Args:
        results: List of BenchmarkResult objects.
        baseline_name: Name of baseline for speedup calculation.

    Returns:
        Markdown-formatted table string.
    """
    # Find baseline for speedup calculation
    baseline_ms = None
    if baseline_name:
        for r in results:
            if baseline_name.lower() in r.name.lower():
                baseline_ms = r.mean_ms
                break

    # Build table
    lines = [
        "| Kernel | Latency (ms) | Bandwidth (GB/s) | TFLOPS | Speedup |",
        "|--------|-------------|------------------|--------|---------|",
    ]

    for r in results:
        speedup = ""
        if baseline_ms and r.mean_ms > 0:
            speedup = f"{baseline_ms / r.mean_ms:.2f}x"

        bw = f"{r.achieved_bandwidth_gb_s:.1f}" if r.achieved_bandwidth_gb_s else "-"
        tflops = f"{r.achieved_tflops:.2f}" if r.achieved_tflops else "-"

        lines.append(
            f"| {r.name} | {r.mean_ms:.3f} ± {r.std_ms:.3f} | {bw} | {tflops} | {speedup} |"
        )

    return "\n".join(lines)


# Convenience function for quick GPU info
def print_gpu_info() -> None:
    """Print GPU information to stdout."""
    try:
        specs = get_gpu_specs()
        print(specs)
    except RuntimeError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Demo: print GPU info
    print_gpu_info()
