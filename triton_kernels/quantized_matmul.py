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


# Compiled dequantization kernel for best performance
@torch.compile(dynamic=True)
def _dequantize_output(y_int32: torch.Tensor, inv_x_scale: torch.Tensor, scale_fp16: torch.Tensor) -> torch.Tensor:
    """Fused dequantization using torch.compile for best performance."""
    return y_int32.half() * inv_x_scale * scale_fp16


def _int8_gemm_cublas(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    scale_fp16: Optional[torch.Tensor] = None,
    act_scale: float = 127.0,
) -> torch.Tensor:
    """
    W8A16 GEMM using cuBLAS INT8 tensor cores (torch._int_mm).

    This is the fast path that uses native INT8 tensor cores (624 TOPS on A100)
    by quantizing activations on-the-fly.

    CRITICAL: weight_int8 must be [N, K] row-major. We pass weight.t() as a
    NON-CONTIGUOUS view to _int_mm, which cuBLAS handles efficiently via
    column-major GEMM. Making it contiguous kills performance (5x slower)!

    Args:
        x: FP16 activations [M, K]
        weight_int8: INT8 weights [N, K] (standard nn.Linear layout)
        scale: FP32 per-channel scales [N]
        scale_fp16: Optional pre-converted FP16 scale [N] (for repeated calls)
        act_scale: Scale factor for activation quantization (default: 127)

    Returns:
        FP16 output [M, N]

    Requirements:
        - M (batch size) must be > 16 for _int_mm

    Performance Notes:
        - INT8 matmul is ~1.5x faster than FP16 (413 vs 243 TFLOPS on A100)
        - But quantization/dequantization overhead can reduce net benefit
        - For best performance, use scale_fp16 parameter to avoid fp32->fp16 conversion
        - torch.compile is used for the dequantization step (~5x faster)
    """
    M, K = x.shape
    N, K2 = weight_int8.shape
    assert K == K2, f"Dimension mismatch: x has K={K}, weight has K={K2}"

    # Quantize activations on-the-fly: FP16 -> INT8
    # Use per-row absmax quantization
    x_absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    inv_x_scale = (x_absmax / act_scale).half()  # [M, 1] - keep inverted for efficiency
    x_int8 = (x * (act_scale / x_absmax)).round().clamp(-128, 127).to(torch.int8)

    # INT8 matmul using cuBLAS tensor cores: [M, K] @ [K, N] -> [M, N] (int32)
    # CRITICAL: Pass weight.t() as a VIEW, not contiguous! cuBLAS is optimized
    # for column-major B matrix (which is what the transpose view gives us).
    # Making it contiguous triggers a 5x slower code path.
    y_int32 = torch._int_mm(x_int8, weight_int8.t())

    # Dequantize: int32 -> fp16 using compiled kernel
    # output = (x_int8 * inv_x_scale) @ (weight_int8 * scale)
    #        = y_int32 * inv_x_scale * scale
    if scale_fp16 is None:
        scale_fp16 = scale.half()
    y_fp16 = _dequantize_output(y_int32, inv_x_scale, scale_fp16)

    return y_fp16


def _int8_gemm_fused(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    weight_transposed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused W8A16 GEMM using Triton with INT8 tensor cores.

    Fuses activation quantization, INT8 matmul, and dequantization into a single kernel.
    This eliminates the overhead of separate kernel launches.

    Args:
        x: FP16 activations [M, K]
        weight_int8: INT8 weights [N, K] (standard nn.Linear layout)
        scale: FP32 per-channel scales [N]
        weight_transposed: Optional pre-transposed weight [K, N]

    Returns:
        FP16 output [M, N]
    """
    M, K = x.shape
    N = weight_int8.shape[0]

    # Pre-compute per-row activation scales: absmax / 127
    # This is a small overhead (~0.05ms) but allows fused kernel
    act_scale = (x.abs().amax(dim=1) / 127.0).clamp(min=1e-6).float()

    # Use pre-transposed weight if available
    if weight_transposed is not None:
        weight_t = weight_transposed
    else:
        weight_t = weight_int8.t().contiguous()

    # Allocate output
    y = torch.empty((M, N), device=x.device, dtype=torch.float16)

    # Grid dimensions
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    # Launch fused kernel
    _int8_gemm_fused_kernel[grid](
        x,
        weight_t,
        y,
        scale,
        act_scale,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        y.stride(0), y.stride(1),
    )

    return y


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _int8_gemm_fused_kernel(
    # Pointers
    A,           # Activation pointer (FP16): [M, K]
    B,           # Weight pointer (INT8): [K, N]
    C,           # Output pointer (FP16): [M, N]
    Scale,       # Weight scale pointer (FP32): [N] (per-output-channel)
    ActScale,    # Activation scale pointer (FP32): [M] (per-row, computed externally)
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
    Fused W8A8 GEMM kernel using INT8 tensor cores.

    This kernel fuses:
    1. Loading FP16 activations
    2. On-the-fly quantization to INT8 using pre-computed per-row scales
    3. INT8 matmul using tensor cores (INT32 accumulation)
    4. Dequantization to FP16 output

    The activation scale (ActScale) must be pre-computed as:
        ActScale[i] = max(|A[i, :]|) / 127.0

    Output is computed as:
        C[i,j] = (sum_k A_int8[i,k] * B[k,j]) * ActScale[i] * Scale[j]
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

    # Load scales for this tile
    # Weight scale: per-channel (indexed by N)
    w_scale = tl.load(Scale + offs_n, mask=offs_n < N, other=1.0)
    # Activation scale: per-row (indexed by M)
    a_scale = tl.load(ActScale + offs_m, mask=offs_m < M, other=1.0)

    # Initialize accumulator in INT32 for INT8 tensor cores
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Main loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k

        # Load A tile (FP16)
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a_fp16 = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Quantize A to INT8 in registers
        # A_int8 = round(A_fp16 / ActScale * 127)
        # Since ActScale = absmax / 127, this becomes: A_int8 = round(A_fp16 * 127 / absmax)
        # We need to scale by 127/ActScale = 127^2/absmax
        inv_scale = 127.0 / (a_scale[:, None] * 127.0 + 1e-6)
        a_int8 = (a_fp16 * inv_scale).to(tl.int8)

        # Load B tile (INT8)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        b_int8 = tl.load(b_ptrs, mask=b_mask, other=0)

        # INT8 matmul with INT32 accumulation
        acc += tl.dot(a_int8, b_int8, out_dtype=tl.int32)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Dequantize output: C = acc * ActScale * Scale
    # The scales are: ActScale[i] (per-row), Scale[j] (per-column)
    c_fp32 = acc.to(tl.float32) * a_scale[:, None] * w_scale[None, :]

    # Write output tile (convert to FP16)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c_fp32.to(tl.float16), mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
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
    Triton kernel for W8A16 GEMM using FP16 tensor cores.

    NOTE: This is the SLOW PATH. For large batches, use _int8_gemm_cublas instead
    which uses native INT8 tensor cores via cuBLAS.

    Computes C = A @ (B * scale) where:
    - A is FP16 activations [M, K]
    - B is INT8 weights [K, N]
    - scale is FP32 per-channel scales [N]
    - C is FP16 output [M, N]

    This kernel dequantizes INT8 weights to FP16 in registers, then uses
    FP16 tensor cores. Benefits:
    - 2x memory bandwidth savings from INT8 weights
    - No activation quantization overhead

    Best for small batch sizes where memory bandwidth dominates.
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

    # Initialize accumulator in FP32 (tensor cores accumulate in FP32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k

        # Load A tile (FP16) - already in correct format for tensor cores
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile (INT8) and dequantize to FP16 in registers
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        b_int8 = tl.load(b_ptrs, mask=b_mask, other=0)

        # Dequantize INT8 -> FP16 (this happens in registers, essentially free)
        # We defer scale multiplication to after accumulation for numerical stability
        b_fp16 = b_int8.to(tl.float16)

        # FP16 matmul using tensor cores with FP32 accumulation
        # This is the key fix: use FP16 inputs to trigger tensor cores
        acc += tl.dot(a, b_fp16, out_dtype=tl.float32)

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
    weight_transposed: Optional[torch.Tensor] = None,
    scale_fp16: Optional[torch.Tensor] = None,
    use_cublas: Optional[bool] = None,
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
        weight_transposed: Optional pre-transposed weight (K, N) to avoid transpose overhead.
        scale_fp16: Optional pre-converted FP16 scale for better performance.
        use_cublas: If True, use cuBLAS INT8 tensor cores.
                    If False, use Triton FP16 dequantization kernel.
                    If None (default), auto-select: cuBLAS for M>16, Triton for M<=16.

    Returns:
        FP16 output tensor of shape (..., N).

    Note:
        Weight is stored as (N, K) but represents the matmul x @ W^T.
        This matches nn.Linear convention where weight is (out_features, in_features).

    Performance:
        - cuBLAS path: Uses INT8 tensor cores (624 TOPS on A100) with on-the-fly
          activation quantization. Requires M > 16.
        - Triton path: Uses FP16 tensor cores (312 TFLOPS on A100) with INT8
          weight dequantization. Works for any batch size.

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
    x_2d = x.view(-1, K).contiguous()
    M = x_2d.shape[0]

    # Auto-select backend based on batch size
    # Default to Triton path for accuracy (no activation quantization error)
    # cuBLAS path is faster but adds activation quantization error (~3-8% extra)
    # Users can explicitly set use_cublas=True for speed over accuracy
    if use_cublas is None:
        use_cublas = False  # Default to accurate Triton path

    if use_cublas and M > 16:
        # Fast path: cuBLAS INT8 tensor cores with on-the-fly activation quantization
        # Pass original weight [N, K], NOT transposed - _int8_gemm_cublas uses weight.t() view
        y = _int8_gemm_cublas(x_2d, weight_int8, scale, scale_fp16)
    else:
        # Triton path: FP16 tensor cores with INT8 weight dequantization
        # Use pre-transposed weight if available, otherwise transpose
        if weight_transposed is not None:
            weight_t = weight_transposed
        else:
            weight_t = weight_int8.t().contiguous()

        y = torch.empty((M, N), device=x.device, dtype=torch.float16)

        # Grid dimensions - let autotune pick the best block sizes
        def grid(META):
            return (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

        # Launch kernel with autotune
        _int8_gemm_kernel[grid](
            x_2d,
            weight_t,
            y,
            scale,
            M, N, K,
            x_2d.stride(0), x_2d.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            y.stride(0), y.stride(1),
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
    Linear layer using INT8 weights with optimized GEMM kernel.

    Uses cuBLAS INT8 tensor cores for large batches (M > 16) and
    Triton FP16 kernel with INT8 weight dequantization for small batches.

    Args:
        in_features: Input dimension (K).
        out_features: Output dimension (N).
        bias: Whether to include bias.

    Performance:
        - Large batches (M > 16): Uses cuBLAS INT8 tensor cores (624 TOPS on A100)
        - Small batches (M <= 16): Uses Triton with FP16 tensor cores
        - Both paths benefit from 2x memory reduction from INT8 weights

    Example:
        >>> linear = Int8Linear(4096, 11008)
        >>> linear.quantize_weights(pretrained_weight)
        >>> linear = linear.cuda()  # Move to GPU
        >>> y = linear(x)  # Uses optimized INT8 GEMM
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
        # Pre-converted FP16 scale for cuBLAS path
        self.register_buffer(
            "scale_fp16",
            torch.ones(out_features, dtype=torch.float16)
        )
        # Pre-transposed weight for Triton path
        self.register_buffer(
            "weight_transposed",
            torch.zeros(in_features, out_features, dtype=torch.int8)
        )

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

        self._quantized = False

    def quantize_weights(self, weight: torch.Tensor) -> None:
        """Quantize and store FP16/FP32 weights as INT8.

        Args:
            weight: FP16 or FP32 weight tensor of shape (out_features, in_features).
                    Can be on any device (will be moved to CPU for quantization).
        """
        assert weight.shape == (self.out_features, self.in_features)

        # Quantization happens on CPU
        weight_cpu = weight.cpu() if weight.is_cuda else weight
        weight_int8, scale = quantize_weight_per_channel(weight_cpu)

        # Move to same device as buffers and copy
        device = self.weight_int8.device
        self.weight_int8.copy_(weight_int8.to(device))
        self.scale.copy_(scale.to(device))

        # Pre-compute derived tensors for performance
        self.scale_fp16.copy_(scale.half().to(device))
        self.weight_transposed.copy_(weight_int8.t().contiguous().to(device))
        self._quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using optimized INT8 GEMM kernel."""
        if not self._quantized:
            raise RuntimeError("Weights not quantized. Call quantize_weights() first.")

        return int8_gemm(
            x, self.weight_int8, self.scale, self.bias,
            weight_transposed=self.weight_transposed,
            scale_fp16=self.scale_fp16,
        )

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear) -> "Int8Linear":
        """Convert nn.Linear to Int8Linear.

        The resulting Int8Linear will be on the same device as the input linear.
        """
        has_bias = linear.bias is not None
        int8_linear = cls(linear.in_features, linear.out_features, bias=has_bias)

        # Move to same device as input linear, then quantize
        device = linear.weight.device
        int8_linear = int8_linear.to(device)
        int8_linear.quantize_weights(linear.weight.data)

        # Copy bias if present
        if has_bias:
            int8_linear.bias.data.copy_(linear.bias.data.half())

        return int8_linear

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quantized={self._quantized}"
        )
