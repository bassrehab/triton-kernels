"""
Quantization utilities for INT8 inference.

Provides symmetric quantization for weights, commonly used in W8A16 inference
where weights are INT8 but activations remain in FP16.

Quantization scheme:
- Symmetric: scale = max(|tensor|) / 127
- quantized = round(tensor / scale).clamp(-128, 127).to(int8)
- dequantized = quantized.float() * scale

This gives ~2x memory reduction for weights while maintaining good accuracy.
The dequantization happens in registers during matmul, so the memory traffic
reduction directly translates to speedup for memory-bound GEMMs.

Reference: LLM.int8() - https://arxiv.org/abs/2208.07339
"""

import torch
from typing import Optional


# TODO: add asymmetric quantization option for better accuracy on skewed distributions
def quantize_symmetric(
    tensor: torch.Tensor,
    bits: int = 8,
    dim: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric quantization of tensor to INT8.

    Computes per-tensor or per-channel quantization using symmetric scheme:
    - scale = max(|tensor|) / (2^(bits-1) - 1)
    - quantized = round(tensor / scale).clamp(-128, 127)

    Args:
        tensor: Input tensor to quantize (typically FP16 or FP32 weights).
        bits: Number of bits for quantization (default: 8).
        dim: Dimension for per-channel quantization. If None, uses per-tensor.
             For weight matrices (out_features, in_features), use dim=0 for
             per-output-channel quantization.

    Returns:
        Tuple of (quantized_int8, scale):
        - quantized_int8: INT8 tensor of same shape as input
        - scale: FP32 scale factor(s) for dequantization

    Example:
        >>> weight = torch.randn(4096, 4096, dtype=torch.float16)
        >>> weight_int8, scale = quantize_symmetric(weight)
        >>> weight_fp16 = dequantize(weight_int8, scale)
        >>> error = (weight - weight_fp16).abs().max()
    """
    assert bits == 8, f"Only 8-bit quantization supported, got {bits}"

    # Max value for symmetric INT8: 127 (we use symmetric range -127 to 127)
    qmax = 127

    # Compute scale
    if dim is None:
        # Per-tensor quantization
        max_val = tensor.abs().max()
        scale = max_val / qmax
    else:
        # Per-channel quantization
        max_val = tensor.abs().amax(dim=dim, keepdim=True)
        scale = max_val / qmax

    # Avoid division by zero
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    # Quantize: round and clamp
    quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)

    # Scale should be FP32 for numerical precision during dequantization
    scale = scale.float()

    # Squeeze scale if per-channel
    if dim is not None:
        scale = scale.squeeze(dim)

    return quantized, scale


def dequantize(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float16,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Dequantize INT8 tensor back to floating point.

    Args:
        quantized: INT8 quantized tensor.
        scale: Scale factor(s) from quantization.
        dtype: Output dtype (default: float16).
        dim: Dimension along which scale was computed (for broadcasting).
             For weight matrices with per-output-channel quantization, use dim=0.

    Returns:
        Dequantized tensor in specified dtype.

    Example:
        >>> weight_int8, scale = quantize_symmetric(weight_fp16)
        >>> weight_restored = dequantize(weight_int8, scale)
    """
    # Convert to float for multiplication
    quantized_float = quantized.float()

    # Handle per-channel scale broadcasting
    if scale.dim() == 1 and quantized.dim() == 2:
        if dim is None or dim == 0:
            # Scale has shape (out_features,), need (out_features, 1)
            scale = scale.unsqueeze(1)
        else:
            # Scale has shape (in_features,), need (1, in_features)
            scale = scale.unsqueeze(0)

    # Dequantize
    dequantized = quantized_float * scale

    return dequantized.to(dtype)


def quantize_weight_per_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weight matrix with per-output-channel scales.

    For a weight matrix of shape (out_features, in_features), computes
    one scale per output channel (row).

    This is the common scheme for W8A16 inference, providing good accuracy
    while allowing efficient dequantization during matmul.

    Args:
        weight: Weight tensor of shape (out_features, in_features).

    Returns:
        Tuple of (weight_int8, scales):
        - weight_int8: INT8 weights of shape (out_features, in_features)
        - scales: FP32 scales of shape (out_features,)

    Example:
        >>> W = torch.randn(4096, 4096, dtype=torch.float16)
        >>> W_int8, scales = quantize_weight_per_channel(W)
    """
    assert weight.dim() == 2, f"Expected 2D weight matrix, got {weight.dim()}D"
    return quantize_symmetric(weight, bits=8, dim=1)


def calculate_quantization_error(
    original: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
    dim: Optional[int] = None,
) -> dict[str, float]:
    """
    Calculate quantization error metrics.

    Args:
        original: Original FP tensor.
        quantized: INT8 quantized tensor.
        scale: Scale factor(s).
        dim: Dimension for scale broadcasting.

    Returns:
        Dictionary with error metrics:
        - max_abs_error: Maximum absolute error
        - mean_abs_error: Mean absolute error
        - relative_error: Max relative error (percentage)
        - snr_db: Signal-to-noise ratio in dB
    """
    # Dequantize
    restored = dequantize(quantized, scale, dtype=original.dtype, dim=dim)

    # Calculate errors
    diff = (original - restored).float()
    original_float = original.float()

    max_abs_error = diff.abs().max().item()
    mean_abs_error = diff.abs().mean().item()

    # Relative error: only consider significant values (> 10% of max) to avoid
    # misleading high errors for small values. For 8-bit quantization, the max
    # relative error is ~scale/(2*value), so small values inherently have high
    # relative error even with perfect quantization.
    threshold = original_float.abs().max() * 0.1
    significant_mask = original_float.abs() > threshold
    if significant_mask.any():
        rel_error = (diff[significant_mask].abs() / original_float[significant_mask].abs()).max().item() * 100
    else:
        rel_error = 0.0

    # Signal-to-noise ratio
    signal_power = (original_float ** 2).mean()
    noise_power = (diff ** 2).mean()
    snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-10)).item()

    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "relative_error_pct": rel_error,
        "snr_db": snr_db,
    }


class QuantizedLinear(torch.nn.Module):
    """
    Linear layer with INT8 quantized weights.

    Stores weights in INT8 format and dequantizes on-the-fly during forward pass.
    This is a reference implementation - the actual kernel-level optimization
    happens in the Triton INT8 GEMM kernel.

    For W8A16 inference:
    - Weights: INT8 (2x memory reduction)
    - Activations: FP16
    - Compute: FP16 with FP32 accumulation

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include bias (default: False for LLM weights).

    Example:
        >>> linear = QuantizedLinear(4096, 4096)
        >>> linear.quantize_weights(pretrained_weight)
        >>> y = linear(x)  # x is FP16, y is FP16
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

        # Register buffers for quantized weights
        self.register_buffer(
            "weight_int8",
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, dtype=torch.float32)
        )

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._quantized = False

    def quantize_weights(self, weight: torch.Tensor) -> None:
        """
        Quantize and store weights.

        Args:
            weight: FP16 or FP32 weight tensor of shape (out_features, in_features).
        """
        assert weight.shape == (self.out_features, self.in_features)
        weight_int8, scale = quantize_weight_per_channel(weight)
        self.weight_int8.copy_(weight_int8)
        self.weight_scale.copy_(scale)
        self._quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with on-the-fly dequantization.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        if not self._quantized:
            raise RuntimeError("Weights not quantized. Call quantize_weights() first.")

        # Dequantize weights
        weight = dequantize(self.weight_int8, self.weight_scale, dtype=x.dtype, dim=0)

        # Compute matmul
        y = torch.nn.functional.linear(x, weight, self.bias)

        return y

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear) -> "QuantizedLinear":
        """
        Create QuantizedLinear from existing nn.Linear.

        Args:
            linear: Pretrained linear layer.

        Returns:
            QuantizedLinear with quantized weights.
        """
        has_bias = linear.bias is not None
        quantized = cls(linear.in_features, linear.out_features, bias=has_bias)
        quantized.quantize_weights(linear.weight.data)
        if has_bias:
            quantized.bias.data.copy_(linear.bias.data)
        return quantized

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quantized={self._quantized}"
        )
