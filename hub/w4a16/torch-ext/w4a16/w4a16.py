"""
W4A16 GEMM: FP16 activations x 4-bit weight-only quantized weights, in Triton.

The dominant LLM inference quantization in 2026 (GPTQ/AWQ) is 4-bit weight-only.
The fast kernels for it (Marlin, AWQ, exllama) are CUDA-only. This is a portable
Triton implementation that runs on NVIDIA and AMD (via the Triton backend).

    y = x @ dequant(W_q)

with x FP16 (M, K), W_q 4-bit packed (K // 8, N int32), and per-group scale /
zero-point along K. Dequantization is fused into the matmul K-loop, so weights
cross the memory bus at 4 bits and are expanded to FP16 on-chip only. This is a
memory-bandwidth win in the decode / small-batch regime (see docs/w4a16.md and
_dev/kernel-roadmap/w4a16/design.md for the roofline).

Weight format (matches reference/w4a16_reference.py):
  - values in [0, 15]; 8 nibbles packed per int32 along K -> packed (K // 8, N);
  - one scale and one zero-point per group of `group_size` rows along K;
  - dequant: w = (q - zero) * scale. Unsigned + zero-point covers both asymmetric
    (per-group zero) and symmetric (fixed zero of 8) quantization.
"""
from typing import Tuple

import torch
import triton
import triton.language as tl


# subhadipmitra, 2026-07-19: batch-size threshold for the split-K decode path (used
# in w4a16_gemm). Split-K wins for small M and loses once there is enough parallelism;
# this crossover is a heuristic to tune on target hardware.
_W4A16_SPLITK_MAX_M = 8


# subhadipmitra, 2026-07-19: configs use BLOCK_K in {32, 64} (multiples of 8, and
# divisors of the supported group sizes 64/128) so each K-tile lies within a single
# quantization group and covers whole packed int32 rows. Small-M configs are first:
# W4A16 is a decode/small-batch kernel where the win is weight bandwidth.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _w4a16_gemm_kernel(
    A,            # Activations (FP16): [M, K]
    B,            # Packed 4-bit weights (int32): [K // 8, N]
    Scales,       # Per-group scales (FP16): [K // GROUP_SIZE, N]
    Zeros,        # Per-group zero-points (FP16): [K // GROUP_SIZE, N]
    C,            # Output (FP16): [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,     # strides of the PACKED weight (rows are K // 8)
    stride_sk, stride_sn,
    stride_zk, stride_zn,
    stride_cm, stride_cn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled GEMM with fused 4-bit unpack + group dequant, FP16 tensor cores, FP32 accum."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_kp = tl.arange(0, BLOCK_K // 8)   # packed rows within a tile (one int32 = 8 K rows)
    shifts = (tl.arange(0, 8) * 4)         # [8] nibble shifts, one per K within an int32

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_kp[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        kk = k + offs_k
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0)

        # subhadipmitra, 2026-07-19: load each packed int32 ONCE (BLOCK_K // 8 rows) and
        # unpack all 8 nibbles in registers via a broadcast shift, then reshape to the
        # (BLOCK_K, BLOCK_N) weight tile. The old kernel indexed with offs_k // 8, which
        # reloaded each int32 8 times and moved ~8x the weight traffic; this is the fix.
        pk_row = k // 8 + offs_kp
        pk = tl.load(b_ptrs, mask=(pk_row[:, None] < (K // 8)) & (offs_n[None, :] < N), other=0)
        b = (pk[:, None, :] >> shifts[None, :, None]) & 0xF   # (BLOCK_K // 8, 8, BLOCK_N)
        b = tl.reshape(b, (BLOCK_K, BLOCK_N)).to(tl.float16)  # row 8*i + j -> K = 8*i + j

        # One scale/zero row per tile: BLOCK_K divides GROUP_SIZE, so the whole tile is
        # in a single group. This replaces the per-element (BLOCK_K, BLOCK_N) scale load.
        gid = k // GROUP_SIZE
        s = tl.load(Scales + gid * stride_sk + offs_n * stride_sn, mask=offs_n < N, other=1.0)
        z = tl.load(Zeros + gid * stride_zk + offs_n * stride_zn, mask=offs_n < N, other=0.0)
        b = (b - z[None, :].to(tl.float16)) * s[None, :].to(tl.float16)

        acc += tl.dot(a, b, out_dtype=tl.float32)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


# subhadipmitra, 2026-07-19: autotune SPLIT_K / BLOCK_N / stages / warps. An A100 sweep
# showed the best split-K config depends on the full shape (attn-qkv wants 8, llama-ffn
# 4, qwen 8 - non-monotonic), so a fixed heuristic regresses some shapes. reset_to_zero
# is required because the kernel accumulates into C with atomic_add: the autotuner must
# re-zero C before each timing trial and the final run, or results are corrupted.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': bn, 'BLOCK_K': 64, 'SPLIT_K': sk},
                      num_stages=ns, num_warps=nw)
        for bn in (64, 128) for sk in (4, 8, 16) for ns in (3, 4) for nw in (2, 4)
    ],
    key=['M', 'N', 'K'],
    reset_to_zero=['C'],
)
@triton.jit
def _w4a16_gemm_splitk_kernel(
    A, B, Scales, Zeros, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_sk, stride_sn,
    stride_zk, stride_zn,
    stride_cm, stride_cn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Split-K W4A16 GEMM for the decode / small-M regime.

    A skinny (small-M) GEMM launches too few programs to saturate HBM, so the plain
    kernel loses to cuBLAS FP16 despite moving 4x less weight data. Splitting the K
    reduction across SPLIT_K programs restores parallelism: each computes a partial
    over a strided K-slice and atomic-adds it into the (pre-zeroed) FP32 output C.
    Measured ~1.2-1.3x over FP16 at M=1 on A100. Unpack/dequant match the main kernel.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_kp = tl.arange(0, BLOCK_K // 8)
    shifts = tl.arange(0, 8) * 4

    a_ptrs = A + offs_m[:, None] * stride_am + (pid_k * BLOCK_K + offs_k)[None, :] * stride_ak
    b_ptrs = B + (pid_k * (BLOCK_K // 8) + offs_kp)[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(pid_k * BLOCK_K, K, SPLIT_K * BLOCK_K):
        kk = k + offs_k
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0)
        pk = tl.load(b_ptrs, mask=(offs_n[None, :] < N), other=0)
        b = (pk[:, None, :] >> shifts[None, :, None]) & 0xF
        b = tl.reshape(b, (BLOCK_K, BLOCK_N)).to(tl.float16)
        gid = k // GROUP_SIZE
        s = tl.load(Scales + gid * stride_sk + offs_n * stride_sn, mask=offs_n < N, other=1.0)
        z = tl.load(Zeros + gid * stride_zk + offs_n * stride_zn, mask=offs_n < N, other=0.0)
        b = (b - z[None, :].to(tl.float16)) * s[None, :].to(tl.float16)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += SPLIT_K * BLOCK_K * stride_ak
        b_ptrs += SPLIT_K * (BLOCK_K // 8) * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.atomic_add(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def w4a16_gemm(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    W4A16 GEMM: FP16 activations x 4-bit weights.

    Computes ``y = x @ dequant(packed)``, dequantizing the 4-bit weights inside the
    kernel so weights cross the memory bus at 4 bits (4x less traffic than FP16).

    Parameters
    ----------
    x : torch.Tensor
        FP16 activations of shape (M, K).
    packed : torch.Tensor
        Packed 4-bit weights, int32, shape (K // 8, N). See `pack_int4`.
    scales : torch.Tensor
        Per-group scales (FP16) of shape (K // group_size, N).
    zeros : torch.Tensor
        Per-group zero-points (FP16) of shape (K // group_size, N).
    group_size : int
        Number of K rows per quantization group.

    Returns
    -------
    y : torch.Tensor
        FP16 output of shape (M, N).
    """
    M, K = x.shape
    N = packed.shape[1]
    if packed.shape[0] != K // 8:
        raise ValueError(f"packed rows ({packed.shape[0]}) must equal K // 8 ({K // 8})")
    if scales.shape != (K // group_size, N) or zeros.shape != (K // group_size, N):
        raise ValueError("scales/zeros must have shape (K // group_size, N)")
    # The hoisted-scale kernel assumes each K-tile lies within a single group, i.e.
    # BLOCK_K divides group_size. Autotune uses BLOCK_K in {32, 64}, so group_size
    # must be a multiple of 64 (covers the common 64 and 128).
    if group_size % 64 != 0:
        raise ValueError(f"group_size ({group_size}) must be a multiple of 64")

    x = x.contiguous()

    # subhadipmitra, 2026-07-19: dispatch on batch size. In the decode / small-M
    # regime the plain kernel launches too few programs to saturate HBM and loses to
    # cuBLAS FP16; split-K adds parallelism there and beats it (~1.2-1.3x at M=1). For
    # larger M the non-split kernel wins (split-K's FP32 atomic output then costs more
    # than it saves).
    if M <= _W4A16_SPLITK_MAX_M:
        # FP32 accumulator (split-K atomic-adds partials); autotune picks the config.
        c = torch.zeros((M, N), device=x.device, dtype=torch.float32)

        def splitk_grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']), meta['SPLIT_K'])

        _w4a16_gemm_splitk_kernel[splitk_grid](
            x, packed, scales, zeros, c,
            M, N, K,
            x.stride(0), x.stride(1),
            packed.stride(0), packed.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            c.stride(0), c.stride(1),
            GROUP_SIZE=group_size,
        )
        return c.to(torch.float16)

    y = torch.empty((M, N), device=x.device, dtype=torch.float16)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    _w4a16_gemm_kernel[grid](
        x, packed, scales, zeros, y,
        M, N, K,
        x.stride(0), x.stride(1),
        packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        y.stride(0), y.stride(1),
        GROUP_SIZE=group_size,
    )
    return y


# ---------------------------------------------------------------------------
# Weight-prep utilities (pure PyTorch, for quantizing a model's weights offline).
# ---------------------------------------------------------------------------

def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack an (K, N) tensor of values in [0, 15] into (K // 8, N) int32.

    Nibble j (bits [4j, 4j+3]) of the int32 holds the value at K = 8*i + j.
    """
    K, N = q.shape
    if K % 8 != 0:
        raise ValueError(f"K ({K}) must be divisible by 8")
    q = q.to(torch.int32).reshape(K // 8, 8, N)
    packed = torch.zeros(K // 8, N, dtype=torch.int32, device=q.device)
    for j in range(8):
        packed |= (q[:, j, :] & 0xF) << (4 * j)
    return packed


def quantize_weight_int4_grouped(
    weight: torch.Tensor,
    group_size: int = 128,
    symmetric: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize an FP weight (K, N) to grouped 4-bit and pack it.

    Returns ``(packed, scales, zeros)`` in the format consumed by `w4a16_gemm`:
    packed int32 (K // 8, N), scales/zeros (K // group_size, N). See the module
    docstring for the numerics.
    """
    K, N = weight.shape
    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")
    if K % 8 != 0:
        raise ValueError(f"K ({K}) must be divisible by 8 for int4 packing")

    w = weight.float().reshape(K // group_size, group_size, N)
    if symmetric:
        scale = (w.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-8)
        zero = torch.full_like(scale, 8.0)
    else:
        wmin = w.amin(dim=1, keepdim=True)
        wmax = w.amax(dim=1, keepdim=True)
        scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)
        zero = (-wmin / scale).round().clamp(0, 15)

    q = (w / scale + zero).round().clamp(0, 15).to(torch.int32).reshape(K, N)
    packed = pack_int4(q)
    return packed, scale.reshape(K // group_size, N).to(weight.dtype), zero.reshape(K // group_size, N).to(weight.dtype)
