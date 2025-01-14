"""
Fused RMSNorm kernel implemented in Triton.

RMSNorm (Root Mean Square Layer Normalization) is used in LLaMA, Mistral, and other
modern LLMs as a simpler alternative to LayerNorm. It normalizes by the RMS of the
input without centering (no mean subtraction).

Formula: y = x * rsqrt(mean(x^2) + eps) * weight

Performance characteristics:
- Memory-bound operation (low arithmetic intensity ~1-2 FLOPs/byte)
- Benefits significantly from fusion with adjacent operations
- Uses FP32 accumulation for numerical stability

Reference: https://arxiv.org/abs/1910.07467
"""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _rmsnorm_kernel(
    X,           # Input tensor pointer
    Y,           # Output tensor pointer
    W,           # Weight tensor pointer
    stride_x,    # Stride for moving between rows in X
    stride_y,    # Stride for moving between rows in Y
    N,           # Hidden dimension (number of elements per row)
    eps,         # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program processes
):
    """
    Triton kernel for RMSNorm.

    Each program instance handles one row of the input tensor.
    The kernel computes: y = x * rsqrt(mean(x^2) + eps) * weight

    Memory access pattern:
    - Each thread block processes one complete row
    - Coalesced reads of x and weight
    - Coalesced write of y
    - FP32 accumulation for variance computation

    Arithmetic intensity:
    - Reads: N elements of x + N elements of weight = 2N * dtype_size
    - Writes: N elements of y = N * dtype_size
    - FLOPs: N (square) + N (sum) + 1 (div) + 1 (add eps) + 1 (rsqrt) + N (mul x) + N (mul weight) ≈ 4N
    - AI ≈ 4N / (3N * 2) = 0.67 for FP16 (memory-bound)
    """
    # Row index - each program handles one row
    row_idx = tl.program_id(0)

    # Compute row start pointers
    X += row_idx * stride_x
    Y += row_idx * stride_y

    # Compute mean of squares using FP32 accumulation for stability
    # Process in blocks to handle arbitrary hidden dimensions
    mean_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # Load input values
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)

        # Accumulate x^2
        mean_sq += x * x

    # Reduce to get sum of squares, then compute mean
    sum_sq = tl.sum(mean_sq, axis=0)
    mean_sq_scalar = sum_sq / N

    # Compute RMS normalization factor: rsqrt(mean(x^2) + eps)
    rrms = tl.rsqrt(mean_sq_scalar + eps)

    # Apply normalization and weight scaling
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # Load input and weight
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

        # Compute output: x * rsqrt(mean(x^2) + eps) * weight
        y = x * rrms * w

        # Store output (cast back to input dtype)
        tl.store(Y + cols, y, mask=mask)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Apply RMS normalization to input tensor.

    Args:
        x: Input tensor of shape (..., hidden_dim). The normalization is applied
           over the last dimension.
        weight: Learnable scale parameter of shape (hidden_dim,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor of same shape as input.

    Example:
        >>> x = torch.randn(2, 1024, 4096, device='cuda', dtype=torch.float16)
        >>> weight = torch.ones(4096, device='cuda', dtype=torch.float16)
        >>> y = rmsnorm(x, weight, eps=1e-6)
    """
    # Validate inputs
    assert x.is_cuda, "Input must be on CUDA device"
    assert weight.is_cuda, "Weight must be on CUDA device"
    assert x.shape[-1] == weight.shape[0], f"Hidden dim mismatch: {x.shape[-1]} vs {weight.shape[0]}"

    # Flatten batch dimensions for kernel launch
    original_shape = x.shape
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    num_rows = x_flat.shape[0]

    # Allocate output
    y_flat = torch.empty_like(x_flat)

    # Choose block size based on hidden dimension
    # Powers of 2 work best, cap at 8192 for register pressure
    # TODO: autotune block size based on GPU architecture
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)

    # Launch kernel: one program per row
    _rmsnorm_kernel[(num_rows,)](
        x_flat,
        y_flat,
        weight,
        x_flat.stride(0),
        y_flat.stride(0),
        hidden_dim,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y_flat.view(original_shape)


def rmsnorm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    PyTorch reference implementation of RMSNorm.

    Used for correctness validation against the Triton kernel.
    """
    # Compute in FP32 for numerical stability
    x_fp32 = x.float()
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * rms).to(x.dtype) * weight


@triton.jit
def _rmsnorm_residual_kernel(
    X,           # Input tensor pointer
    R,           # Residual tensor pointer
    Y,           # Output tensor pointer
    W,           # Weight tensor pointer
    stride_x,    # Stride for moving between rows in X
    stride_r,    # Stride for moving between rows in R
    stride_y,    # Stride for moving between rows in Y
    N,           # Hidden dimension (number of elements per row)
    eps,         # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program processes
):
    """
    Triton kernel for fused RMSNorm with residual addition.

    Computes: y = RMSNorm(x + residual) * weight

    This fusion saves one memory round-trip compared to separate operations:
    - Naive: read x, read residual, write temp; read temp, write y (4N transfers)
    - Fused: read x, read residual, write y (3N transfers)

    For memory-bound operations, this gives ~1.3-1.5x speedup.

    Arithmetic intensity (fused):
    - Reads: N (x) + N (residual) + N (weight) = 3N * dtype_size
    - Writes: N (y) = N * dtype_size
    - FLOPs: N (add) + N (square) + N (sum) + 1 (div) + 1 (add eps) + 1 (rsqrt) + N (mul x) + N (mul weight) ≈ 5N
    - AI ≈ 5N / (4N * 2) = 0.625 for FP16 (memory-bound)
    """
    # Row index - each program handles one row
    row_idx = tl.program_id(0)

    # Compute row start pointers
    X += row_idx * stride_x
    R += row_idx * stride_r
    Y += row_idx * stride_y

    # Compute mean of squares of (x + residual) using FP32 accumulation
    mean_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # Load input and residual
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask=mask, other=0.0).to(tl.float32)

        # Fused add
        x_plus_r = x + r

        # Accumulate (x + residual)^2
        mean_sq += x_plus_r * x_plus_r

    # Reduce to get sum of squares, then compute mean
    sum_sq = tl.sum(mean_sq, axis=0)
    mean_sq_scalar = sum_sq / N

    # Compute RMS normalization factor
    rrms = tl.rsqrt(mean_sq_scalar + eps)

    # Apply normalization and weight scaling
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # Load input, residual, and weight
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

        # Compute fused output: (x + residual) * rsqrt(mean((x+r)^2) + eps) * weight
        x_plus_r = x + r
        y = x_plus_r * rrms * w

        # Store output
        tl.store(Y + cols, y, mask=mask)


def rmsnorm_residual_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused RMSNorm with residual addition.

    Computes: y = RMSNorm(x + residual) * weight

    This is more efficient than separate add + rmsnorm because it avoids
    materializing the intermediate (x + residual) tensor in global memory.

    Args:
        x: Input tensor of shape (..., hidden_dim).
        residual: Residual tensor of same shape as x.
        weight: Learnable scale parameter of shape (hidden_dim,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor of same shape as input.

    Example:
        >>> x = torch.randn(2, 1024, 4096, device='cuda', dtype=torch.float16)
        >>> residual = torch.randn_like(x)
        >>> weight = torch.ones(4096, device='cuda', dtype=torch.float16)
        >>> y = rmsnorm_residual_fused(x, residual, weight, eps=1e-6)
    """
    # Validate inputs
    assert x.is_cuda, "Input must be on CUDA device"
    assert residual.is_cuda, "Residual must be on CUDA device"
    assert weight.is_cuda, "Weight must be on CUDA device"
    assert x.shape == residual.shape, f"Shape mismatch: x={x.shape}, residual={residual.shape}"
    assert x.shape[-1] == weight.shape[0], f"Hidden dim mismatch: {x.shape[-1]} vs {weight.shape[0]}"

    # Flatten batch dimensions for kernel launch
    original_shape = x.shape
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    residual_flat = residual.view(-1, hidden_dim)
    num_rows = x_flat.shape[0]

    # Allocate output
    y_flat = torch.empty_like(x_flat)

    # Choose block size based on hidden dimension
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)

    # Launch kernel: one program per row
    _rmsnorm_residual_kernel[(num_rows,)](
        x_flat,
        residual_flat,
        y_flat,
        weight,
        x_flat.stride(0),
        residual_flat.stride(0),
        y_flat.stride(0),
        hidden_dim,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y_flat.view(original_shape)


def rmsnorm_residual_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    PyTorch reference implementation of fused RMSNorm + residual.

    Used for correctness validation.
    """
    x_plus_r = (x + residual).float()
    rms = torch.rsqrt(x_plus_r.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_plus_r * rms).to(x.dtype) * weight


class TritonRMSNorm(torch.nn.Module):
    """
    RMSNorm layer using Triton kernel.

    Drop-in replacement for torch.nn.RMSNorm with identical interface.

    Args:
        hidden_size: Size of the last dimension to normalize over.
        eps: Small constant for numerical stability.

    Example:
        >>> norm = TritonRMSNorm(4096).cuda()
        >>> x = torch.randn(2, 1024, 4096, device='cuda', dtype=torch.float16)
        >>> y = norm(x)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"
