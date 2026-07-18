"""
PyTorch reference for W4A16 GEMM: 4-bit weight-only quantized matmul.

Ground truth for the Triton `w4a16_gemm` kernel. Everything here is plain PyTorch
(no Triton), so it runs on CPU and defines the exact numerics the kernel must match.

Format (GPTQ/AWQ style, group-wise along K):
  - weights are quantized to 4 bits (unsigned, values in [0, 15]) with one scale and
    one zero-point per group of `group_size` rows along K;
  - 8 nibbles are packed into one int32 along K, so packed weights are (K // 8, N);
  - dequantization is  w = (q - zero) * scale.

The unsigned [0, 15] + zero-point representation covers both asymmetric quantization
(per-group zero-point) and symmetric quantization (fixed zero-point of 8).
"""
from typing import Tuple

import torch


def quantize_weight_int4_grouped(
    weight: torch.Tensor,
    group_size: int = 128,
    symmetric: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize an FP weight matrix to grouped 4-bit and pack it.

    Parameters
    ----------
    weight : torch.Tensor
        Weight of shape (K, N) (contraction dim first, matching `x @ weight`).
    group_size : int
        Number of rows along K that share a scale/zero-point. Must divide K.
    symmetric : bool
        If True, use a fixed zero-point of 8 (symmetric around zero). If False,
        use a per-group asymmetric zero-point (GPTQ/AWQ style).

    Returns
    -------
    packed : torch.Tensor
        Packed 4-bit weights, int32, shape (K // 8, N). 8 nibbles per int32 along K.
    scales : torch.Tensor
        Per-group scales, shape (K // group_size, N), same dtype as `weight`.
    zeros : torch.Tensor
        Per-group zero-points (as float), shape (K // group_size, N).
    """
    K, N = weight.shape
    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")
    if K % 8 != 0:
        raise ValueError(f"K ({K}) must be divisible by 8 for int4 packing")

    w = weight.float().reshape(K // group_size, group_size, N)  # (G, gs, N)

    if symmetric:
        max_abs = w.abs().amax(dim=1, keepdim=True)  # (G, 1, N)
        scale = (max_abs / 7.0).clamp(min=1e-8)
        zero = torch.full_like(scale, 8.0)
    else:
        wmin = w.amin(dim=1, keepdim=True)
        wmax = w.amax(dim=1, keepdim=True)
        scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)
        # zero-point that maps wmin -> 0 in quantized space
        zero = (-wmin / scale).round().clamp(0, 15)

    q = (w / scale + zero).round().clamp(0, 15).to(torch.int32)  # (G, gs, N) in [0,15]
    q = q.reshape(K, N)

    packed = pack_int4(q)
    scales = scale.reshape(K // group_size, N).to(weight.dtype)
    zeros = zero.reshape(K // group_size, N).to(weight.dtype)
    return packed, scales, zeros


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack an (K, N) tensor of values in [0, 15] into (K // 8, N) int32.

    Nibble j of the int32 (bits [4j, 4j+3]) holds the value at K = 8*i + j.
    """
    K, N = q.shape
    if K % 8 != 0:
        raise ValueError(f"K ({K}) must be divisible by 8")
    q = q.to(torch.int32).reshape(K // 8, 8, N)
    packed = torch.zeros(K // 8, N, dtype=torch.int32, device=q.device)
    for j in range(8):
        packed |= (q[:, j, :] & 0xF) << (4 * j)
    return packed


def unpack_int4(packed: torch.Tensor, K: int) -> torch.Tensor:
    """Inverse of `pack_int4`: (K // 8, N) int32 -> (K, N) int32 in [0, 15]."""
    K8, N = packed.shape
    out = torch.zeros(K, N, dtype=torch.int32, device=packed.device)
    for j in range(8):
        out[j::8][: K8] = (packed >> (4 * j)) & 0xF
    return out


def dequantize_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    K: int,
) -> torch.Tensor:
    """Reconstruct the (K, N) FP weight from packed 4-bit + group scale/zero."""
    q = unpack_int4(packed, K).float()                       # (K, N)
    g = torch.arange(K, device=packed.device) // group_size  # (K,) group index per row
    s = scales.float()[g]                                    # (K, N)
    z = zeros.float()[g]                                     # (K, N)
    return ((q - z) * s)


def w4a16_reference(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Reference W4A16 GEMM: y = x @ dequant(packed).

    Parameters
    ----------
    x : torch.Tensor
        Activations, shape (M, K).
    packed : torch.Tensor
        Packed 4-bit weights, int32, shape (K // 8, N).
    scales, zeros : torch.Tensor
        Per-group scale / zero-point, shape (K // group_size, N).
    group_size : int
        Group size along K.

    Returns
    -------
    y : torch.Tensor
        Output of shape (M, N), same dtype as `x`.
    """
    K = x.shape[1]
    w = dequantize_int4(packed, scales, zeros, group_size, K)  # (K, N), float32
    # Accumulate in float32 (matches the kernel's FP32 accumulation), then cast back.
    return (x.float() @ w).to(x.dtype)
