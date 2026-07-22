"""Roofline analysis for the W4A16 GEMM kernel.

Sweeps batch size M from decode (1) to prefill for real projection shapes, benchmarks
the W4A16 kernel and a torch FP16 matmul, and plots both on the A100 roofline. Requires
a CUDA GPU (Triton JIT-compiles at runtime).

    python benchmarks/roofline/w4a16_roofline.py --shape qwen2-72b-ffn \
        --output docs/figures/w4a16_roofline.png
"""
import argparse

import torch
import triton

from benchmarks.utils import get_gpu_specs, plot_roofline, BenchmarkResult
from triton_kernels.w4a16 import w4a16_gemm, quantize_weight_int4_grouped

# (K, N) projection shapes.
SHAPES = {
    "llama3-8b-attn-qkv": (4096, 4096),
    "llama3-8b-ffn-up": (4096, 14336),
    "qwen2-72b-ffn": (8192, 29568),
    # subhadipmitra, 2026-07-19: current-generation shapes (models released within the
    # last two years). Dims track the rows above closely; the W4A16 win follows the
    # matrix shape, not the model's release date.
    "llama3.3-70b-ffn": (8192, 28672),     # Llama 3.3 70B (Dec 2024), gate/up proj
    "qwen3-32b-ffn": (5120, 25600),        # Qwen3-32B (2025), gate/up proj
    "deepseek-v3.2-ffn": (7168, 18432),    # DeepSeek-V3.2 (2026), dense MLP gate/up
    "deepseek-v3.2-moe-up": (7168, 2048),  # DeepSeek-V3.2 (2026), MoE expert up (256 experts)
}


def _profile(K: int, N: int, M: int, group_size: int, w4a16: bool) -> BenchmarkResult:
    dev = "cuda"
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    W = torch.randn(K, N) * 0.1
    flops = 2.0 * M * K * N
    act, out = M * K * 2, M * N * 2
    if w4a16:
        packed, scales, zeros = quantize_weight_int4_grouped(W, group_size, symmetric=False)
        packed, scales, zeros = packed.to(dev), scales.to(dev).half(), zeros.to(dev).half()
        ms = triton.testing.do_bench(lambda: w4a16_gemm(x, packed, scales, zeros, group_size))
        wbytes = K * N * 0.5 + (K // group_size) * N * 2 * 2
        name = f"W4A16 M={M}"
    else:
        w_fp16 = W.to(dev).half()
        ms = triton.testing.do_bench(lambda: x @ w_fp16)
        wbytes = K * N * 2
        name = f"FP16 M={M}"
    return BenchmarkResult(name=name, mean_ms=ms, std_ms=0.0, min_ms=ms, max_ms=ms,
                           flops=flops, bytes_accessed=wbytes + act + out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="qwen2-72b-ffn", choices=list(SHAPES))
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--batch-sizes", default="1,32,128")
    ap.add_argument("--output", default=None, help="PNG path (e.g. docs/figures/w4a16_roofline.png)")
    args = ap.parse_args()

    specs = get_gpu_specs()
    print(specs)
    K, N = SHAPES[args.shape]
    results = []
    for M in (int(m) for m in args.batch_sizes.split(",")):
        for w4 in (True, False):
            r = _profile(K, N, M, args.group_size, w4)
            print(f"  {r}")
            results.append(r)

    plot_roofline(
        results, specs,
        title=f"W4A16 vs FP16 GEMM Roofline - {args.shape} (K={K}, N={N})",
        save_path=args.output, show=args.output is None,
    )


if __name__ == "__main__":
    main()
