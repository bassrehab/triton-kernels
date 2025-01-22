"""
INT8 Quantized Matrix Multiplication (W8A16 GEMM) implemented in Triton.

W8A16 means: Weights are INT8, Activations are FP16.

This is the common inference quantization scheme where:
1. Weights are pre-quantized to INT8 (2x memory reduction)
2. Activations remain in FP16 for accuracy
3. Dequantization happens in registers during the matmul
4. Accumulation uses FP32 for precision

The key insight is that weight loading is often the bottleneck in LLM inference.
By storing weights in INT8:
- Memory traffic is reduced by 2x
- Dequantization is essentially "free" (happens in fast registers)
- Net effect: ~1.5-2x speedup for memory-bound GEMMs

Memory access pattern:
- Load INT8 weights: 1 byte per element
- Load FP16 activations: 2 bytes per element
- Dequantize weights to FP16 in registers
- Compute FP16 matmul with FP32 accumulation
- Store FP16 result: 2 bytes per element

Reference: LLM.int8() - https://arxiv.org/abs/2208.07339
"""

import torch
import triton
import triton.language as tl
from typing import Optional

from triton_kernels.quantization import quantize_weight_per_channel, dequantize


@triton.jit
def _int8_gemm_kernel(
    # Pointers
    A,           # Activation pointer (FP16): [M, K]
    B,           # Weight pointer (INT8): [K, N]
    C,           # Output pointer (FP16): [M, N]
    Scale,       # Scale pointer (FP32): [N] (per-output-channel)
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for W8A16 GEMM.

    Computes C = A @ (B * scale) where:
    - A is FP16 activations [M, K]
    - B is INT8 weights [K, N]
    - scale is FP32 per-channel scales [N]
    - C is FP16 output [M, N]

    Memory access pattern:
    - Tiled computation with BLOCK_M x BLOCK_N output tiles
    - Each thread block computes one output tile
    - Accumulation in FP32 for numerical precision
    - Dequantization of INT8 weights happens in registers

    The key optimization is that INT8 weights are 2x smaller than FP16,
    reducing memory traffic for the weight-bound matmul.
    """
    # Program ID for this tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A and B tiles
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Load scale for this output tile (per-channel, so indexed by N)
    scale = tl.load(Scale + offs_n, mask=offs_n < N, other=1.0)

    # Initialize accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k

        # Load A tile (FP16)
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile (INT8) and dequantize to FP16
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        b_int8 = tl.load(b_ptrs, mask=b_mask, other=0)

        # Dequantize: convert INT8 to FP32, then multiply by scale
        # Note: We defer the scale multiplication to after accumulation
        # for numerical stability
        b_fp32 = b_int8.to(tl.float32)

        # Accumulate: acc += a @ b
        # Convert a to FP32 for accumulation
        acc += tl.dot(a.to(tl.float32), b_fp32)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply scale (per-output-channel)
    acc = acc * scale[None, :]

    # Write output tile (convert to FP16)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def int8_gemm(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    W8A16 GEMM: FP16 activations x INT8 weights.

    Computes: y = x @ (weight_int8 * scale) + bias

    The weight dequantization happens inside the kernel, so memory traffic
    for weights is reduced by 2x compared to FP16 weights.

    Args:
        x: FP16 activation tensor of shape (..., K).
        weight_int8: INT8 weight tensor of shape (N, K) - note: transposed!
        scale: FP32 scale tensor of shape (N,) for per-output-channel dequant.
        bias: Optional FP16 bias of shape (N,).

    Returns:
        FP16 output tensor of shape (..., N).

    Note:
        Weight is stored as (N, K) but represents the matmul x @ W^T.
        This matches nn.Linear convention where weight is (out_features, in_features).

    Example:
        >>> x = torch.randn(1, 2048, 4096, device='cuda', dtype=torch.float16)
        >>> weight_int8, scale = quantize_weight_per_channel(
        ...     torch.randn(11008, 4096, dtype=torch.float16)
        ... )
        >>> y = int8_gemm(x, weight_int8.cuda(), scale.cuda())
        >>> print(y.shape)  # (1, 2048, 11008)
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert weight_int8.is_cuda, "Weight must be on CUDA"
    assert scale.is_cuda, "Scale must be on CUDA"
    assert weight_int8.dtype == torch.int8, "Weight must be INT8"
    assert x.dtype == torch.float16, "Input must be FP16"

    # Get dimensions
    # x: (..., K), weight: (N, K) -> output: (..., N)
    original_shape = x.shape
    K = x.shape[-1]
    N = weight_int8.shape[0]

    assert weight_int8.shape == (N, K), f"Weight shape mismatch: {weight_int8.shape} vs ({N}, {K})"
    assert scale.shape == (N,), f"Scale shape mismatch: {scale.shape} vs ({N},)"

    # Reshape x to 2D for matmul
    x_2d = x.view(-1, K)
    M = x_2d.shape[0]

    # Transpose weight for efficient memory access: (N, K) -> (K, N)
    weight_t = weight_int8.t().contiguous()

    # Allocate output
    y = torch.empty((M, N), device=x.device, dtype=torch.float16)

    # Choose block sizes
    # TODO: autotune these for different matrix shapes and GPU types
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Grid dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    _int8_gemm_kernel[grid](
        x_2d,
        weight_t,
        y,
        scale,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Add bias if present
    if bias is not None:
        y = y + bias

    # Reshape to original batch dimensions
    output_shape = original_shape[:-1] + (N,)
    return y.view(output_shape)


def int8_gemm_torch(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PyTorch reference implementation of W8A16 GEMM.

    Used for correctness validation.
    """
    # Dequantize weight
    weight_fp16 = dequantize(weight_int8, scale, dtype=torch.float16, dim=0)

    # Standard matmul
    y = torch.nn.functional.linear(x, weight_fp16, bias)

    return y


class Int8Linear(torch.nn.Module):
    """
    Linear layer using INT8 weights with Triton GEMM kernel.

    This is the optimized version that uses the Triton kernel for the matmul,
    providing ~1.5-2x speedup over FP16 for memory-bound GEMMs.

    Args:
        in_features: Input dimension (K).
        out_features: Output dimension (N).
        bias: Whether to include bias.

    Example:
        >>> linear = Int8Linear(4096, 11008)
        >>> linear.quantize_weights(pretrained_weight)
        >>> y = linear(x)  # Uses Triton INT8 GEMM
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # INT8 weights: (out_features, in_features)
        self.register_buffer(
            "weight_int8",
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer(
            "scale",
            torch.ones(out_features, dtype=torch.float32)
        )

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

        self._quantized = False

    def quantize_weights(self, weight: torch.Tensor) -> None:
        """Quantize and store FP16/FP32 weights as INT8."""
        assert weight.shape == (self.out_features, self.in_features)
        weight_int8, scale = quantize_weight_per_channel(weight)
        self.weight_int8.copy_(weight_int8)
        self.scale.copy_(scale)
        self._quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using INT8 GEMM kernel."""
        if not self._quantized:
            raise RuntimeError("Weights not quantized")

        return int8_gemm(x, self.weight_int8, self.scale, self.bias)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear) -> "Int8Linear":
        """Convert nn.Linear to Int8Linear."""
        has_bias = linear.bias is not None
        int8_linear = cls(linear.in_features, linear.out_features, bias=has_bias)
        int8_linear.quantize_weights(linear.weight.data)
        if has_bias:
            int8_linear.bias.data.copy_(linear.bias.data.half())
        return int8_linear

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quantized={self._quantized}"
        )
