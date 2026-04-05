"""
Roofline analysis for MoE dispatch kernel stages.

Profiles each stage of the MoE pipeline separately to identify bottlenecks
and measure how close each stage operates to its theoretical ceiling.

Stages analyzed:
1. Router (compute-bound for large batches)
2. Permute (memory-bound — pure data movement)
3. Expert GEMM / Fused gate+up (compute-bound)
4. Unpermute + combine (memory-bound)

Generates a roofline plot showing each stage's operational intensity vs
achieved performance, overlaid on the GPU's memory bandwidth and compute ceilings.

Usage:
    python benchmarks/roofline/moe_roofline.py --model mixtral-8x7b --output docs/figures/moe_roofline.png
"""

import argparse
import torch
import triton
from typing import Optional

from benchmarks.utils import (
    get_gpu_specs,
    benchmark_fn,
    plot_roofline,
    BenchmarkResult,
    GPUSpecs,
)


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


def profile_router(num_tokens: int, cfg: dict) -> BenchmarkResult:
    """Profile the router stage: matmul + softmax + top-k."""
    from triton_kernels.moe.router import moe_router

    hidden_dim = cfg["hidden_dim"]
    num_experts = cfg["num_experts"]
    top_k = cfg["top_k"]

    x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
    router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float16)

    # FLOPs: matmul (2*M*N*K) + softmax (~5*M*N) + top-k (~K*M*top_k)
    flops = 2.0 * num_tokens * hidden_dim * num_experts + 5.0 * num_tokens * num_experts
    # Bytes: read x + read weights + write logits + write indices + write weights
    dtype_size = 2
    bytes_acc = (num_tokens * hidden_dim + num_experts * hidden_dim) * dtype_size  # reads
    bytes_acc += num_tokens * (num_experts + top_k * 2) * dtype_size  # writes

    return benchmark_fn(
        lambda: moe_router(x, router_weight, top_k),
        name="Router",
        flops=flops,
        bytes_accessed=bytes_acc,
    )


def profile_permute(num_tokens: int, cfg: dict) -> BenchmarkResult:
    """Profile the permute stage: token reordering to expert-contiguous layout."""
    from triton_kernels.moe.permute import permute_tokens

    hidden_dim = cfg["hidden_dim"]
    num_experts = cfg["num_experts"]
    top_k = cfg["top_k"]

    x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
    indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")

    # subhadipmitra, 2026-03-21: permute is pure data movement, ~0 FLOPs.
    # Using a small FLOPs value so roofline plot places it correctly in
    # the memory-bound region
    flops = float(num_tokens * top_k)  # trivial index arithmetic
    dtype_size = 2
    bytes_acc = num_tokens * hidden_dim * dtype_size  # read input
    bytes_acc += num_tokens * top_k * hidden_dim * dtype_size  # write permuted

    return benchmark_fn(
        lambda: permute_tokens(x, indices, num_experts),
        name="Permute",
        flops=flops,
        bytes_accessed=bytes_acc,
    )


def profile_expert_gemm_unfused(num_tokens: int, cfg: dict) -> BenchmarkResult:
    """Profile the expert FFN stage (unfused: 3 separate grouped GEMMs)."""
    from triton_kernels.moe.permute import permute_tokens
    from triton_kernels.moe.expert_gemm import expert_ffn_triton

    hidden_dim = cfg["hidden_dim"]
    ffn_dim = cfg["ffn_dim"]
    num_experts = cfg["num_experts"]
    top_k = cfg["top_k"]

    x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
    indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")
    w_gate = torch.randn(num_experts, ffn_dim, hidden_dim, device="cuda", dtype=torch.float16) * 0.02
    w_up = torch.randn(num_experts, ffn_dim, hidden_dim, device="cuda", dtype=torch.float16) * 0.02
    w_down = torch.randn(num_experts, hidden_dim, ffn_dim, device="cuda", dtype=torch.float16) * 0.02

    permuted, offsets, _, _ = permute_tokens(x, indices, num_experts)

    total_tokens = num_tokens * top_k
    # FLOPs: gate + up + down GEMMs + SiLU+mul
    flops = 2.0 * total_tokens * hidden_dim * ffn_dim  # gate
    flops += 2.0 * total_tokens * hidden_dim * ffn_dim  # up
    flops += 2.0 * total_tokens * ffn_dim * hidden_dim  # down
    flops += 3.0 * total_tokens * ffn_dim  # SiLU + mul

    dtype_size = 2
    # Bytes: read tokens + read all weights + read/write intermediates + write output
    bytes_acc = total_tokens * hidden_dim * dtype_size  # read permuted tokens (x3 for 3 GEMMs, but cached)
    bytes_acc += num_experts * (2 * ffn_dim * hidden_dim + hidden_dim * ffn_dim) * dtype_size  # weights
    bytes_acc += total_tokens * ffn_dim * dtype_size * 4  # gate_out + up_out write + read + intermediate write
    bytes_acc += total_tokens * hidden_dim * dtype_size  # output

    return benchmark_fn(
        lambda: expert_ffn_triton(permuted, w_gate, w_up, w_down, offsets, num_experts),
        name="Expert FFN (unfused)",
        flops=flops,
        bytes_accessed=bytes_acc,
    )


def profile_expert_gemm_fused(num_tokens: int, cfg: dict) -> BenchmarkResult:
    """Profile the fused expert FFN (fused gate+up + regular down)."""
    from triton_kernels.moe.permute import permute_tokens
    from triton_kernels.moe.fused_moe import fused_expert_ffn

    hidden_dim = cfg["hidden_dim"]
    ffn_dim = cfg["ffn_dim"]
    num_experts = cfg["num_experts"]
    top_k = cfg["top_k"]

    x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
    indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")
    top_k_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device="cuda", dtype=torch.float32), dim=-1
    ).half()

    w_gate = torch.randn(num_experts, ffn_dim, hidden_dim, device="cuda", dtype=torch.float16) * 0.02
    w_up = torch.randn(num_experts, ffn_dim, hidden_dim, device="cuda", dtype=torch.float16) * 0.02
    w_down = torch.randn(num_experts, hidden_dim, ffn_dim, device="cuda", dtype=torch.float16) * 0.02

    permuted, offsets, sorted_idx, _ = permute_tokens(x, indices, num_experts)

    total_tokens = num_tokens * top_k
    flops = 2.0 * total_tokens * hidden_dim * ffn_dim * 2  # gate + up (fused)
    flops += 2.0 * total_tokens * ffn_dim * hidden_dim  # down
    flops += 3.0 * total_tokens * ffn_dim  # SiLU + mul

    dtype_size = 2
    # subhadipmitra, 2026-03-21: fused kernel reads tokens once for gate+up,
    # saving one full read of (total_tokens * hidden_dim). Also eliminates
    # gate_out and up_out write+read
    bytes_acc = total_tokens * hidden_dim * dtype_size  # read permuted (once, fused)
    bytes_acc += num_experts * (2 * ffn_dim * hidden_dim + hidden_dim * ffn_dim) * dtype_size  # weights
    bytes_acc += total_tokens * ffn_dim * dtype_size * 2  # intermediate write + read
    bytes_acc += total_tokens * hidden_dim * dtype_size  # expert output + unpermute

    return benchmark_fn(
        lambda: fused_expert_ffn(
            permuted, w_gate, w_up, w_down, offsets, num_experts,
            top_k_weights, sorted_idx, num_tokens, top_k,
        ),
        name="Expert FFN (fused)",
        flops=flops,
        bytes_accessed=bytes_acc,
    )


def profile_unpermute(num_tokens: int, cfg: dict) -> BenchmarkResult:
    """Profile the unpermute + weighted combine stage."""
    from triton_kernels.moe.permute import permute_tokens, unpermute_tokens

    hidden_dim = cfg["hidden_dim"]
    num_experts = cfg["num_experts"]
    top_k = cfg["top_k"]

    x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
    indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")
    weights = torch.softmax(
        torch.randn(num_tokens, top_k, device="cuda", dtype=torch.float32), dim=-1
    ).half()

    _, _, _, restore_idx = permute_tokens(x, indices, num_experts)
    expert_out = torch.randn(num_tokens * top_k, hidden_dim, device="cuda", dtype=torch.float16)

    # FLOPs: weight * output + accumulate across top_k
    flops = 2.0 * num_tokens * top_k * hidden_dim  # mul + add
    dtype_size = 2
    bytes_acc = num_tokens * top_k * hidden_dim * dtype_size  # read expert outputs
    bytes_acc += num_tokens * top_k * dtype_size  # read weights
    bytes_acc += num_tokens * hidden_dim * dtype_size  # write combined output

    return benchmark_fn(
        lambda: unpermute_tokens(expert_out, weights, restore_idx, num_tokens, top_k),
        name="Unpermute+Combine",
        flops=flops,
        bytes_accessed=bytes_acc,
    )


def run_roofline(
    model_name: str,
    num_tokens: int = 512,
    output_path: Optional[str] = None,
):
    """Run roofline analysis for all MoE stages."""
    cfg = MODEL_CONFIGS[model_name]
    gpu = get_gpu_specs()

    print(f"\n{'='*70}")
    print(f"MoE Roofline Analysis: {model_name} ({num_tokens} tokens)")
    print(f"GPU: {gpu.name}")
    print(f"{'='*70}\n")

    results = []

    print("Profiling stages...")
    for name, profile_fn in [
        ("Router", profile_router),
        ("Permute", profile_permute),
        ("Expert FFN (unfused)", profile_expert_gemm_unfused),
        ("Expert FFN (fused)", profile_expert_gemm_fused),
        ("Unpermute+Combine", profile_unpermute),
    ]:
        try:
            r = profile_fn(num_tokens, cfg)
            results.append(r)
            bw = r.achieved_bandwidth_gb_s or 0
            tflops = r.achieved_tflops or 0
            ai = r.arithmetic_intensity or 0
            efficiency_bw = (bw / gpu.memory_bandwidth_gb_s * 100) if bw > 0 else 0
            efficiency_compute = (tflops / gpu.fp16_tflops * 100) if tflops > 0 else 0
            print(f"  {name:30s}  {r.mean_ms:8.3f} ms  "
                  f"AI={ai:7.2f}  BW={bw:7.1f} GB/s ({efficiency_bw:4.1f}%)  "
                  f"TFLOPS={tflops:6.2f} ({efficiency_compute:4.1f}%)")
        except Exception as e:
            print(f"  {name:30s}  FAILED: {e}")

    # Generate roofline plot
    if output_path:
        try:
            plot_roofline(
                results,
                gpu_specs=gpu,
                title=f"MoE Dispatch Roofline — {model_name} ({num_tokens} tokens)",
                save_path=output_path,
                show=False,
            )
            print(f"\nRoofline plot saved to: {output_path}")
        except Exception as e:
            print(f"\nFailed to generate plot: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MoE Roofline Analysis")
    parser.add_argument("--model", type=str, default="mixtral-8x7b",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save roofline plot (e.g. docs/figures/moe_roofline.png)")
    parser.add_argument("--all", action="store_true",
                        help="Run for all model configs")
    args = parser.parse_args()

    if args.all:
        for model_name in MODEL_CONFIGS:
            suffix = model_name.replace("-", "_")
            output = args.output.replace(".png", f"_{suffix}.png") if args.output else None
            run_roofline(model_name, args.num_tokens, output)
    else:
        run_roofline(args.model, args.num_tokens, args.output)


if __name__ == "__main__":
    main()
