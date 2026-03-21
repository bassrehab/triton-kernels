"""
Benchmarks for MoE dispatch kernels.

Compares:
1. PyTorch reference MoE (loop-over-experts baseline)
2. Triton unfused pipeline (router + permute + grouped GEMM + unpermute)
3. Triton fused pipeline (router + permute + fused gate+up + grouped down + unpermute)

Measures latency, tokens/sec, and achieved bandwidth across model configurations:
- Mixtral-8x7B:   8 experts, top-2, hidden=4096, ffn=14336
- Mixtral-8x22B:  8 experts, top-2, hidden=6144, ffn=16384
- DeepSeek-V3:    256 experts, top-8, hidden=7168, ffn=2048
- Qwen2-MoE-57B:  64 experts, top-4, hidden=3584, ffn=2560

Usage:
    python benchmarks/bench_moe_dispatch.py --model mixtral-8x7b --batch-sizes 1,32,128,512,2048,4096
    python benchmarks/bench_moe_dispatch.py --all
"""

import argparse
import sys
import torch
import triton
from typing import Optional

from benchmarks.utils import (
    get_gpu_specs,
    benchmark_fn,
    format_results_table,
    BenchmarkResult,
)


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "mixtral-8x7b": {
        "num_experts": 8, "top_k": 2,
        "hidden_dim": 4096, "ffn_dim": 14336,
    },
    "mixtral-8x22b": {
        "num_experts": 8, "top_k": 2,
        "hidden_dim": 6144, "ffn_dim": 16384,
    },
    "deepseek-v3": {
        "num_experts": 256, "top_k": 8,
        "hidden_dim": 7168, "ffn_dim": 2048,
    },
    "qwen2-moe-57b": {
        "num_experts": 64, "top_k": 4,
        "hidden_dim": 3584, "ffn_dim": 2560,
    },
}


def calculate_moe_metrics(
    num_tokens: int,
    hidden_dim: int,
    ffn_dim: int,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype = torch.float16,
) -> tuple:
    """
    Calculate FLOPs and bytes for a full MoE forward pass.

    Returns
    -------
    flops : float
        Total floating point operations.
    bytes_accessed : float
        Total bytes read + written (approximate).
    """
    dtype_size = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    # subhadipmitra, 2026-03-21: FLOPs breakdown:
    # Router: num_tokens * hidden_dim * num_experts * 2 (matmul)
    # Gate GEMM: num_tokens * top_k * hidden_dim * ffn_dim * 2
    # Up GEMM: same as gate
    # Down GEMM: num_tokens * top_k * ffn_dim * hidden_dim * 2
    # SiLU + mul: num_tokens * top_k * ffn_dim * 3 (sigmoid + mul + mul)
    router_flops = 2.0 * num_tokens * hidden_dim * num_experts
    gate_flops = 2.0 * num_tokens * top_k * hidden_dim * ffn_dim
    up_flops = gate_flops
    down_flops = 2.0 * num_tokens * top_k * ffn_dim * hidden_dim
    activation_flops = 3.0 * num_tokens * top_k * ffn_dim
    total_flops = router_flops + gate_flops + up_flops + down_flops + activation_flops

    # Bytes: input/output tokens + expert weights (read once per batch)
    # subhadipmitra, 2026-03-21: for large batch sizes the weight reads dominate,
    # for small batches the token reads dominate. This is a rough estimate
    token_bytes = num_tokens * hidden_dim * dtype_size * 2  # read input + write output
    permute_bytes = num_tokens * top_k * hidden_dim * dtype_size * 2  # permute + unpermute
    weight_bytes = num_experts * (2 * ffn_dim * hidden_dim + hidden_dim * ffn_dim) * dtype_size  # gate+up+down
    intermediate_bytes = num_tokens * top_k * ffn_dim * dtype_size * 2  # write + read intermediate
    total_bytes = token_bytes + permute_bytes + weight_bytes + intermediate_bytes

    return total_flops, total_bytes


def _create_moe_inputs(num_tokens: int, cfg: dict, device: str = "cuda"):
    """Create random inputs for MoE benchmarking."""
    hidden_dim = cfg["hidden_dim"]
    ffn_dim = cfg["ffn_dim"]
    num_experts = cfg["num_experts"]

    x = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.float16)
    router_weight = torch.randn(num_experts, hidden_dim, device=device, dtype=torch.float16) * 0.02
    w_gate = torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=torch.float16) * 0.02
    w_up = torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=torch.float16) * 0.02
    w_down = torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=torch.float16) * 0.02

    return x, router_weight, w_gate, w_up, w_down


def bench_pytorch_reference(num_tokens: int, cfg: dict) -> BenchmarkResult:
    """Benchmark PyTorch reference MoE (loop-over-experts)."""
    from reference.moe_reference import MoEReference

    hidden_dim = cfg["hidden_dim"]
    ffn_dim = cfg["ffn_dim"]
    num_experts = cfg["num_experts"]
    top_k = cfg["top_k"]

    model = MoEReference(hidden_dim, ffn_dim, num_experts, top_k).cuda().half()
    x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)

    flops, bytes_acc = calculate_moe_metrics(num_tokens, hidden_dim, ffn_dim, num_experts, top_k)

    return benchmark_fn(
        lambda: model(x),
        name=f"PyTorch Reference",
        flops=flops,
        bytes_accessed=bytes_acc,
    )


def bench_triton_unfused(num_tokens: int, cfg: dict) -> BenchmarkResult:
    """Benchmark Triton unfused pipeline."""
    from triton_kernels.moe.router import moe_router
    from triton_kernels.moe.permute import permute_tokens, unpermute_tokens
    from triton_kernels.moe.expert_gemm import expert_ffn_triton

    hidden_dim = cfg["hidden_dim"]
    ffn_dim = cfg["ffn_dim"]
    num_experts = cfg["num_experts"]
    top_k = cfg["top_k"]

    x, router_weight, w_gate, w_up, w_down = _create_moe_inputs(num_tokens, cfg)

    flops, bytes_acc = calculate_moe_metrics(num_tokens, hidden_dim, ffn_dim, num_experts, top_k)

    def run_unfused():
        indices, weights, _ = moe_router(x, router_weight, top_k)
        permuted, offsets, sorted_idx, restore_idx = permute_tokens(x, indices, num_experts)
        expert_out = expert_ffn_triton(permuted, w_gate, w_up, w_down, offsets, num_experts)
        return unpermute_tokens(expert_out, weights, restore_idx, num_tokens, top_k)

    return benchmark_fn(
        run_unfused,
        name=f"Triton Unfused",
        flops=flops,
        bytes_accessed=bytes_acc,
    )


def bench_triton_fused(num_tokens: int, cfg: dict) -> BenchmarkResult:
    """Benchmark Triton fused pipeline."""
    from triton_kernels.moe.fused_moe import fused_moe_forward

    hidden_dim = cfg["hidden_dim"]
    ffn_dim = cfg["ffn_dim"]
    num_experts = cfg["num_experts"]
    top_k = cfg["top_k"]

    x, router_weight, w_gate, w_up, w_down = _create_moe_inputs(num_tokens, cfg)

    flops, bytes_acc = calculate_moe_metrics(num_tokens, hidden_dim, ffn_dim, num_experts, top_k)

    def run_fused():
        return fused_moe_forward(x, router_weight, w_gate, w_up, w_down, num_experts, top_k)

    return benchmark_fn(
        run_fused,
        name=f"Triton Fused",
        flops=flops,
        bytes_accessed=bytes_acc,
    )


def run_benchmark(
    model_name: str,
    batch_sizes: list,
    skip_reference: bool = False,
):
    """Run full benchmark suite for a model configuration."""
    cfg = MODEL_CONFIGS[model_name]
    gpu = get_gpu_specs()

    print(f"\n{'='*70}")
    print(f"MoE Dispatch Benchmark: {model_name}")
    print(f"{'='*70}")
    print(f"GPU: {gpu.name}")
    print(f"Config: {cfg['num_experts']} experts, top-{cfg['top_k']}, "
          f"hidden={cfg['hidden_dim']}, ffn={cfg['ffn_dim']}")
    print(f"{'='*70}\n")

    all_results = []

    for num_tokens in batch_sizes:
        print(f"\n--- {num_tokens} tokens ---")
        results = []

        # subhadipmitra, 2026-03-21: skip reference for large token counts
        # on DeepSeek-V3 — 256 experts * 3 matmuls * Python loop is painfully slow
        if not skip_reference and not (model_name == "deepseek-v3" and num_tokens > 128):
            try:
                r = bench_pytorch_reference(num_tokens, cfg)
                results.append(r)
                print(f"  {r}")
            except Exception as e:
                print(f"  PyTorch Reference: FAILED ({e})")

        try:
            r = bench_triton_unfused(num_tokens, cfg)
            results.append(r)
            print(f"  {r}")
        except Exception as e:
            print(f"  Triton Unfused: FAILED ({e})")

        try:
            r = bench_triton_fused(num_tokens, cfg)
            results.append(r)
            print(f"  {r}")
        except Exception as e:
            print(f"  Triton Fused: FAILED ({e})")

        all_results.extend(results)

    # Print summary table
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    baseline_name = "PyTorch" if not skip_reference else "Unfused"
    print(format_results_table(all_results, baseline_name=baseline_name))

    return all_results


def main():
    parser = argparse.ArgumentParser(description="MoE Dispatch Benchmark")
    parser.add_argument("--model", type=str, default="mixtral-8x7b",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model configuration to benchmark")
    parser.add_argument("--batch-sizes", type=str, default="1,32,128,512,2048",
                        help="Comma-separated list of batch sizes (num_tokens)")
    parser.add_argument("--all", action="store_true",
                        help="Benchmark all model configurations")
    parser.add_argument("--skip-reference", action="store_true",
                        help="Skip PyTorch reference (faster)")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    if args.all:
        for model_name in MODEL_CONFIGS:
            run_benchmark(model_name, batch_sizes, args.skip_reference)
    else:
        run_benchmark(args.model, batch_sizes, args.skip_reference)


if __name__ == "__main__":
    main()
