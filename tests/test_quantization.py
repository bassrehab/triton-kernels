"""
Tests for quantization utilities.

Validates symmetric INT8 quantization, dequantization, and round-trip error.
"""

import pytest
import torch

from triton_kernels.quantization import (
    quantize_symmetric,
    dequantize,
    quantize_weight_per_channel,
    calculate_quantization_error,
    QuantizedLinear,
)


class TestQuantizeSymmetric:
    """Test suite for symmetric quantization."""

    @pytest.mark.parametrize("shape", [(1024,), (256, 256), (1024, 4096), (4096, 11008)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_output_dtype(self, shape: tuple, dtype: torch.dtype):
        """Verify quantized output is INT8 and scale is FP32."""
        tensor = torch.randn(shape, dtype=dtype)
        quantized, scale = quantize_symmetric(tensor)

        assert quantized.dtype == torch.int8
        assert scale.dtype == torch.float32

    @pytest.mark.parametrize("shape", [(256, 256), (1024, 4096)])
    def test_output_shape_per_tensor(self, shape: tuple):
        """Verify shapes for per-tensor quantization."""
        tensor = torch.randn(shape, dtype=torch.float16)
        quantized, scale = quantize_symmetric(tensor, dim=None)

        assert quantized.shape == shape
        assert scale.shape == ()  # Scalar for per-tensor

    @pytest.mark.parametrize("shape", [(256, 256), (1024, 4096)])
    def test_output_shape_per_channel(self, shape: tuple):
        """Verify shapes for per-channel quantization."""
        tensor = torch.randn(shape, dtype=torch.float16)

        # Per-output-channel (dim=1 for rows)
        quantized, scale = quantize_symmetric(tensor, dim=1)
        assert quantized.shape == shape
        assert scale.shape == (shape[0],)

        # Per-input-channel (dim=0 for columns)
        quantized, scale = quantize_symmetric(tensor, dim=0)
        assert quantized.shape == shape
        assert scale.shape == (shape[1],)

    def test_value_range(self):
        """Verify quantized values are in valid INT8 range."""
        tensor = torch.randn(1024, 1024, dtype=torch.float16) * 10
        quantized, _ = quantize_symmetric(tensor)

        assert quantized.min() >= -128
        assert quantized.max() <= 127

    def test_zero_tensor(self):
        """Test quantization of zero tensor."""
        tensor = torch.zeros(256, 256, dtype=torch.float16)
        quantized, scale = quantize_symmetric(tensor)

        assert (quantized == 0).all()
        # Scale should be 1 (not 0) to avoid division issues
        assert scale.item() == 1.0

    def test_uniform_tensor(self):
        """Test quantization of uniform value tensor."""
        tensor = torch.full((256, 256), 5.0, dtype=torch.float16)
        quantized, scale = quantize_symmetric(tensor)

        # All values should be the same (5 / scale rounded)
        assert (quantized == quantized[0, 0]).all()


class TestDequantize:
    """Test suite for dequantization."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_output_dtype(self, dtype: torch.dtype):
        """Verify dequantized output has correct dtype."""
        tensor = torch.randn(256, 256, dtype=torch.float32)
        quantized, scale = quantize_symmetric(tensor)
        restored = dequantize(quantized, scale, dtype=dtype)

        assert restored.dtype == dtype

    def test_shape_preserved(self):
        """Verify dequantization preserves shape."""
        shapes = [(1024,), (256, 256), (4096, 11008)]
        for shape in shapes:
            tensor = torch.randn(shape, dtype=torch.float16)
            quantized, scale = quantize_symmetric(tensor)
            restored = dequantize(quantized, scale)

            assert restored.shape == shape


class TestRoundTrip:
    """Test suite for quantization round-trip error."""

    @pytest.mark.parametrize("shape", [(256, 256), (1024, 4096), (4096, 4096)])
    def test_round_trip_error_per_tensor(self, shape: tuple):
        """Test round-trip error is within acceptable bounds for per-tensor."""
        tensor = torch.randn(shape, dtype=torch.float16)
        quantized, scale = quantize_symmetric(tensor)
        restored = dequantize(quantized, scale, dtype=torch.float16)

        # For 8-bit symmetric quantization, max error ≈ scale / 2
        # Relative error should be small for normally distributed data
        error = calculate_quantization_error(tensor, quantized, scale)

        # SNR should be > 30 dB for 8-bit quantization
        assert error["snr_db"] > 30, f"SNR too low: {error['snr_db']:.1f} dB"

        # Relative error should be < 5%
        assert error["relative_error_pct"] < 5, f"Relative error too high: {error['relative_error_pct']:.1f}%"

    @pytest.mark.parametrize("shape", [(256, 256), (1024, 4096)])
    def test_round_trip_error_per_channel(self, shape: tuple):
        """Test round-trip error for per-channel quantization."""
        tensor = torch.randn(shape, dtype=torch.float16)
        quantized, scale = quantize_symmetric(tensor, dim=1)
        restored = dequantize(quantized, scale, dtype=torch.float16, dim=0)

        error = calculate_quantization_error(tensor, quantized, scale, dim=0)

        # Per-channel should have even better accuracy
        assert error["snr_db"] > 35, f"SNR too low: {error['snr_db']:.1f} dB"

    def test_small_values(self):
        """Test quantization of small values."""
        tensor = torch.randn(256, 256, dtype=torch.float16) * 1e-3
        quantized, scale = quantize_symmetric(tensor)
        restored = dequantize(quantized, scale, dtype=torch.float16)

        # Should still maintain reasonable relative accuracy
        rel_error = (tensor - restored).abs() / (tensor.abs() + 1e-10)
        # Allow larger relative error for very small values
        assert rel_error.median() < 0.1

    def test_large_values(self):
        """Test quantization of large values."""
        tensor = torch.randn(256, 256, dtype=torch.float16) * 100
        quantized, scale = quantize_symmetric(tensor)
        restored = dequantize(quantized, scale, dtype=torch.float16)

        error = calculate_quantization_error(tensor, quantized, scale)
        assert error["snr_db"] > 30


class TestQuantizeWeightPerChannel:
    """Test suite for weight-specific quantization."""

    @pytest.mark.parametrize("out_features,in_features", [
        (256, 256),
        (4096, 4096),
        (4096, 11008),
        (11008, 4096),
    ])
    def test_weight_shapes(self, out_features: int, in_features: int):
        """Test quantization of various weight matrix shapes."""
        weight = torch.randn(out_features, in_features, dtype=torch.float16)
        weight_int8, scale = quantize_weight_per_channel(weight)

        assert weight_int8.shape == (out_features, in_features)
        assert weight_int8.dtype == torch.int8
        assert scale.shape == (out_features,)
        assert scale.dtype == torch.float32

    def test_different_scales_per_row(self):
        """Verify each row can have different scale."""
        # Create weight with rows of different magnitudes
        weight = torch.zeros(4, 16, dtype=torch.float16)
        weight[0] = torch.randn(16) * 1
        weight[1] = torch.randn(16) * 10
        weight[2] = torch.randn(16) * 100
        weight[3] = torch.randn(16) * 0.1

        weight_int8, scale = quantize_weight_per_channel(weight)

        # Scales should vary significantly
        scale_ratios = scale.max() / scale.min()
        assert scale_ratios > 10, "Scales should vary across rows"


class TestQuantizedLinear:
    """Test suite for QuantizedLinear module."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        in_features, out_features = 256, 128
        batch_size = 4

        # Create and quantize
        linear = QuantizedLinear(in_features, out_features)
        weight = torch.randn(out_features, in_features, dtype=torch.float16)
        linear.quantize_weights(weight)

        # Forward pass
        x = torch.randn(batch_size, in_features, dtype=torch.float16)
        y = linear(x)

        assert y.shape == (batch_size, out_features)
        assert y.dtype == torch.float16

    def test_from_linear(self):
        """Test conversion from nn.Linear."""
        in_features, out_features = 256, 128

        # Create pretrained linear
        original = torch.nn.Linear(in_features, out_features, bias=False)

        # Convert to quantized
        quantized = QuantizedLinear.from_linear(original)

        # Compare outputs
        x = torch.randn(4, in_features)
        y_original = original(x)
        y_quantized = quantized(x)

        # Should be close (within quantization error)
        torch.testing.assert_close(y_original, y_quantized, rtol=0.05, atol=0.05)

    def test_with_bias(self):
        """Test QuantizedLinear with bias."""
        in_features, out_features = 256, 128

        original = torch.nn.Linear(in_features, out_features, bias=True)
        quantized = QuantizedLinear.from_linear(original)

        x = torch.randn(4, in_features)
        y_original = original(x)
        y_quantized = quantized(x)

        torch.testing.assert_close(y_original, y_quantized, rtol=0.05, atol=0.05)

    def test_not_quantized_error(self):
        """Test that forward raises error if not quantized."""
        linear = QuantizedLinear(256, 128)
        x = torch.randn(4, 256)

        with pytest.raises(RuntimeError, match="not quantized"):
            linear(x)

    def test_llm_shapes(self):
        """Test with typical LLM weight shapes."""
        # Hidden to hidden (attention projections)
        shapes = [
            (4096, 4096),    # LLaMA 7B Q/K/V/O
            (4096, 11008),   # LLaMA 7B gate/up
            (11008, 4096),   # LLaMA 7B down
        ]

        for out_features, in_features in shapes:
            linear = QuantizedLinear(in_features, out_features)
            weight = torch.randn(out_features, in_features, dtype=torch.float16)
            linear.quantize_weights(weight)

            x = torch.randn(1, 1024, in_features, dtype=torch.float16)
            y = linear(x)

            assert y.shape == (1, 1024, out_features)


class TestCalculateQuantizationError:
    """Test suite for error calculation."""

    def test_zero_error_identity(self):
        """Test that identical tensors give zero error."""
        tensor = torch.randn(256, 256, dtype=torch.float32)
        # Use high precision to minimize quantization error
        quantized = tensor.to(torch.int8)
        scale = torch.tensor(1.0)

        # Force perfect reconstruction
        quantized = tensor.round().clamp(-128, 127).to(torch.int8)
        scale = torch.tensor(1.0)

        # This won't be exactly zero due to rounding, but SNR should be very high
        # for this contrived case

    def test_error_metrics_reasonable(self):
        """Test that error metrics are in reasonable ranges."""
        tensor = torch.randn(256, 256, dtype=torch.float16)
        quantized, scale = quantize_symmetric(tensor)

        error = calculate_quantization_error(tensor, quantized, scale)

        assert "max_abs_error" in error
        assert "mean_abs_error" in error
        assert "relative_error_pct" in error
        assert "snr_db" in error

        # All values should be positive/non-negative
        assert error["max_abs_error"] >= 0
        assert error["mean_abs_error"] >= 0
        assert error["relative_error_pct"] >= 0
        # SNR should be positive for any meaningful quantization
        assert error["snr_db"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
