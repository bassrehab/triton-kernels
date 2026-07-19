"""Benchmark W4A16 GEMM vs torch FP16 matmul.

Shows the two things the design predicts (see _dev/kernel-roadmap/w4a16/design.md):
  - a decode / small-batch speedup from 4x less weight traffic, and
  - achieved weight-stream bandwidth approaching the HBM peak in that regime,
    degrading to parity with FP16 by prefill.

Reports, per shape and per batch size M: W4A16 latency, torch-FP16 latency, the
speedup, and the achieved bandwidth on the 4-bit weight stream.
"""
import argparse

import torch
import triton

from triton_kernels.w4a16 import w4a16_gemm, quantize_weight_int4_grouped

# NVIDIA A100-SXM4-80GB peak HBM bandwidth (GB/s).
PEAK_BW_GBS = 2039.0

# (K, N) projection shapes from real models.
SHAPES = {
    "llama3-8b-attn-qkv": (4096, 4096),
    "llama3-8b-ffn-up": (4096, 14336),
    "qwen2-72b-ffn": (8192, 29568),
}
M_SWEEP = [1, 8, 32, 128, 512, 2048, 4096]


def bench_case(K: int, N: int, M: int, group_size: int, dev: str = "cuda"):
    torch.manual_seed(0)
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    W = torch.randn(K, N) * 0.1
    packed, scales, zeros = quantize_weight_int4_grouped(W, group_size, symmetric=False)
    packed = packed.to(dev)
    scales = scales.to(dev).half()
    zeros = zeros.to(dev).half()
    w_fp16 = W.to(dev).half()

    t_w4 = triton.testing.do_bench(lambda: w4a16_gemm(x, packed, scales, zeros, group_size))
    t_fp = triton.testing.do_bench(lambda: x @ w_fp16)

    # Weight-stream bytes: packed 4-bit + FP16 scales + FP16 zeros.
    w4_bytes = K * N * 0.5 + (K // group_size) * N * 2 * 2
    bw_w4 = w4_bytes / (t_w4 * 1e-3) / 1e9  # GB/s
    return t_w4, t_fp, t_fp / t_w4, bw_w4, 100.0 * bw_w4 / PEAK_BW_GBS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--shapes", nargs="*", default=list(SHAPES))
    ap.add_argument("--batch-sizes", type=str, default=",".join(map(str, M_SWEEP)))
    args = ap.parse_args()
    ms = [int(m) for m in args.batch_sizes.split(",")]

    print(f"W4A16 vs torch FP16  |  group_size={args.group_size}  |  peak HBM {PEAK_BW_GBS:.0f} GB/s")
    for name in args.shapes:
        K, N = SHAPES[name]
        print(f"\n{name}  (K={K}, N={N})")
        print(f"  {'M':>5} {'W4A16(ms)':>11} {'FP16(ms)':>10} {'speedup':>9} {'W4A16 BW':>11} {'%peak':>7}")
        for M in ms:
            t_w4, t_fp, sp, bw, pct = bench_case(K, N, M, args.group_size)
            print(f"  {M:>5} {t_w4:>11.3f} {t_fp:>10.3f} {sp:>8.2f}x {bw:>9.0f}GB {pct:>6.0f}%")


if __name__ == "__main__":
    main()
