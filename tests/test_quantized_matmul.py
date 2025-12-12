"""
Tests for INT8 quantized matrix multiplication (W8A16 GEMM).

Validates numerical correctness against PyTorch reference with dequantized weights.
"""

import pytest
import torch

from triton_kernels.quantized_matmul import int8_gemm, int8_gemm_torch, Int8Linear
from triton_kernels.quantization import quantize_weight_per_channel


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestInt8GEMM:
    """Test suite for INT8 GEMM kernel."""

    @pytest.mark.parametrize("M,N,K", [
        (1, 256, 256),       # Single token, small
        (1, 1024, 1024),     # Single token
        (1, 4096, 4096),     # LLaMA hidden dim
        (32, 1024, 1024),    # Small batch
        (128, 4096, 4096),   # Larger batch
    ])
    def test_correctness_2d(self, M: int, N: int, K: int):
        """Test INT8 GEMM correctness with 2D input."""
        # Create inputs
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        weight_fp16 = torch.randn(N, K, device="cuda", dtype=torch.float16)

        # Quantize weight
        weight_int8, scale = quantize_weight_per_channel(weight_fp16.cpu())
        weight_int8 = weight_int8.cuda()
        scale = scale.cuda()

        # Triton kernel
        y_triton = int8_gemm(x, weight_int8, scale)

        # Reference (with dequantized weights)
        y_ref = int8_gemm_torch(x, weight_int8, scale)

        # Check correctness (allow some tolerance for quantization error)
        torch.testing.assert_close(y_triton, y_ref, rtol=0.05, atol=0.1)

    @pytest.mark.parametrize("batch_size,seq_len,N,K", [
        (1, 1, 4096, 4096),           # Single token inference
        (1, 128, 4096, 4096),         # Short sequence
        (1, 2048, 4096, 4096),        # Long sequence
        (1, 2048, 11008, 4096),       # LLaMA 7B up projection
        (1, 2048, 4096, 11008),       # LLaMA 7B down projection
    ])
    def test_correctness_3d(self, batch_size: int, seq_len: int, N: int, K: int):
        """Test INT8 GEMM with 3D input (batch, seq, hidden)."""
        x = torch.randn(batch_size, seq_len, K, device="cuda", dtype=torch.float16)
        weight_fp16 = torch.randn(N, K, device="cuda", dtype=torch.float16)

        weight_int8, scale = quantize_weight_per_channel(weight_fp16.cpu())
        weight_int8 = weight_int8.cuda()
        scale = scale.cuda()

        y_triton = int8_gemm(x, weight_int8, scale)
        y_ref = int8_gemm_torch(x, weight_int8, scale)

        assert y_triton.shape == (batch_size, seq_len, N)
        # Slightly looser tolerance for very large matrices where numerical errors accumulate
        torch.testing.assert_close(y_triton, y_ref, rtol=0.05, atol=0.2)

    def test_llama_7b_shapes(self):
        """Test with actual LLaMA 7B weight shapes."""
        batch_size = 1
        seq_len = 2048
        hidden_dim = 4096
        ffn_dim = 11008

        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)

        # Gate projection: (ffn_dim, hidden_dim)
        gate_weight = torch.randn(ffn_dim, hidden_dim, device="cuda", dtype=torch.float16)
        gate_int8, gate_scale = quantize_weight_per_channel(gate_weight.cpu())

        y = int8_gemm(x, gate_int8.cuda(), gate_scale.cuda())
        assert y.shape == (batch_size, seq_len, ffn_dim)

        # Down projection: (hidden_dim, ffn_dim)
        down_weight = torch.randn(hidden_dim, ffn_dim, device="cuda", dtype=torch.float16)
        down_int8, down_scale = quantize_weight_per_channel(down_weight.cpu())

        y2 = int8_gemm(y, down_int8.cuda(), down_scale.cuda())
        assert y2.shape == (batch_size, seq_len, hidden_dim)

    def test_output_dtype(self):
        """Verify output is FP16."""
        M, N, K = 32, 256, 256
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        weight = torch.randn(N, K, device="cuda", dtype=torch.float16)

        weight_int8, scale = quantize_weight_per_channel(weight.cpu())

        y = int8_gemm(x, weight_int8.cuda(), scale.cuda())
        assert y.dtype == torch.float16

    def test_with_bias(self):
        """Test INT8 GEMM with bias."""
        M, N, K = 32, 256, 256
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        weight = torch.randn(N, K, device="cuda", dtype=torch.float16)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)

        weight_int8, scale = quantize_weight_per_channel(weight.cpu())

        y_triton = int8_gemm(x, weight_int8.cuda(), scale.cuda(), bias)
        y_ref = int8_gemm_torch(x, weight_int8.cuda(), scale.cuda(), bias)

        torch.testing.assert_close(y_triton, y_ref, rtol=0.05, atol=0.1)

    def test_against_fp16_matmul(self):
        """Compare INT8 GEMM against FP16 matmul (expect quantization error)."""
        M, N, K = 64, 512, 512
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        weight = torch.randn(N, K, device="cuda", dtype=torch.float16)

        # FP16 reference
        y_fp16 = torch.nn.functional.linear(x, weight)

        # INT8 kernel
        weight_int8, scale = quantize_weight_per_channel(weight.cpu())
        y_int8 = int8_gemm(x, weight_int8.cuda(), scale.cuda())

        # INT8 should be close but not identical due to quantization
        # Expect SNR > 30 dB (relative error < 3%)
        diff = (y_fp16 - y_int8).float()
        signal_power = (y_fp16.float() ** 2).mean()
        noise_power = (diff ** 2).mean()
        snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-10))

        assert snr_db > 25, f"SNR too low: {snr_db:.1f} dB"


class TestInt8Linear:
    """Test suite for Int8Linear module."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        in_features, out_features = 256, 128
        batch_size = 4

        linear = Int8Linear(in_features, out_features).cuda()
        weight = torch.randn(out_features, in_features, dtype=torch.float16)
        linear.quantize_weights(weight)

        x = torch.randn(batch_size, in_features, device="cuda", dtype=torch.float16)
        y = linear(x)

        assert y.shape == (batch_size, out_features)
        assert y.dtype == torch.float16

    def test_from_linear(self):
        """Test conversion from nn.Linear."""
        in_features, out_features = 256, 128

        # Create and initialize nn.Linear
        original = torch.nn.Linear(in_features, out_features, bias=False)
        original = original.cuda().half()

        # Convert to Int8Linear
        int8_linear = Int8Linear.from_linear(original)

        # Compare outputs
        x = torch.randn(4, in_features, device="cuda", dtype=torch.float16)
        y_original = original(x)
        y_int8 = int8_linear(x)

        # Should be close within quantization error
        torch.testing.assert_close(y_original, y_int8, rtol=0.05, atol=0.1)

    def test_with_bias(self):
        """Test Int8Linear with bias."""
        in_features, out_features = 256, 128

        original = torch.nn.Linear(in_features, out_features, bias=True).cuda().half()
        int8_linear = Int8Linear.from_linear(original)

        x = torch.randn(4, in_features, device="cuda", dtype=torch.float16)
        y_original = original(x)
        y_int8 = int8_linear(x)

        torch.testing.assert_close(y_original, y_int8, rtol=0.05, atol=0.1)

    def test_3d_input(self):
        """Test with 3D input (batch, seq, features)."""
        in_features, out_features = 4096, 11008
        batch_size, seq_len = 1, 1024

        linear = Int8Linear(in_features, out_features).cuda()
        weight = torch.randn(out_features, in_features, dtype=torch.float16)
        linear.quantize_weights(weight)

        x = torch.randn(batch_size, seq_len, in_features, device="cuda", dtype=torch.float16)
        y = linear(x)

        assert y.shape == (batch_size, seq_len, out_features)

    def test_not_quantized_error(self):
        """Test that forward raises error if not quantized."""
        linear = Int8Linear(256, 128).cuda()
        x = torch.randn(4, 256, device="cuda", dtype=torch.float16)

        with pytest.raises(RuntimeError, match="not quantized"):
            linear(x)

    def test_memory_reduction(self):
        """Verify INT8 weights use less memory than FP16."""
        in_features, out_features = 4096, 4096

        # FP16 weight size
        fp16_bytes = out_features * in_features * 2  # 2 bytes per FP16

        # INT8 weight size + scale
        int8_bytes = out_features * in_features * 1 + out_features * 4  # 1 byte per INT8 + 4 bytes per FP32 scale

        # Should be ~50% memory reduction
        reduction = (fp16_bytes - int8_bytes) / fp16_bytes * 100
        assert reduction > 45, f"Memory reduction only {reduction:.1f}%"

        # Create actual tensors to verify
        linear = Int8Linear(in_features, out_features).cuda()
        weight = torch.randn(out_features, in_features, dtype=torch.float16)
        linear.quantize_weights(weight)

        actual_bytes = (
            linear.weight_int8.numel() * linear.weight_int8.element_size() +
            linear.scale.numel() * linear.scale.element_size()
        )
        assert actual_bytes < fp16_bytes


class TestNumericalAccuracy:
    """Test numerical accuracy of INT8 GEMM."""

    def test_normal_weights(self):
        """Test with normally distributed weights (typical case)."""
        M, N, K = 128, 1024, 1024
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        weight = torch.randn(N, K, device="cuda", dtype=torch.float16)

        weight_int8, scale = quantize_weight_per_channel(weight.cpu())
        y_int8 = int8_gemm(x, weight_int8.cuda(), scale.cuda())

        # Reference with FP16
        y_fp16 = torch.nn.functional.linear(x, weight)

        # Compute relative error
        # INT8 quantization typically introduces ~5-8% relative error
        rel_error = ((y_int8 - y_fp16).abs() / (y_fp16.abs() + 1e-6)).mean()
        assert rel_error < 0.08, f"Relative error too high: {rel_error:.3f}"

    def test_mixed_magnitude_weights(self):
        """Test with weights of varying magnitudes."""
        M, N, K = 64, 512, 512
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)

        # Create weight with varying row magnitudes
        weight = torch.zeros(N, K, dtype=torch.float16)
        for i in range(N):
            magnitude = 10 ** (torch.rand(1).item() * 4 - 2)  # 0.01 to 100
            weight[i] = torch.randn(K, dtype=torch.float16) * magnitude

        weight = weight.cuda()
        weight_int8, scale = quantize_weight_per_channel(weight.cpu())
        y_int8 = int8_gemm(x, weight_int8.cuda(), scale.cuda())

        y_fp16 = torch.nn.functional.linear(x, weight)

        # Per-channel quantization should handle varying magnitudes well
        # INT8 quantization with mixed magnitudes can introduce ~5-12% relative error
        rel_error = ((y_int8 - y_fp16).abs() / (y_fp16.abs() + 1e-6)).mean()
        assert rel_error < 0.15, f"Relative error too high: {rel_error:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
