"""
Tests for SwiGLU Triton kernel.

Validates numerical correctness against PyTorch reference implementation.
Tests various input shapes, dtypes, and edge cases.
"""

import pytest
import torch
import torch.nn.functional as F

from triton_kernels.swiglu import swiglu_fused, swiglu_torch, SwiGLU


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestSwiGLU:
    """Test suite for SwiGLU kernel."""

    @pytest.mark.parametrize("size", [64, 128, 256, 512, 1024, 2048, 4096, 11008])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    def test_correctness_1d(self, size: int, dtype: torch.dtype):
        """Test SwiGLU on 1D input."""
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("BF16 not supported on this GPU")

        gate = torch.randn(size, device="cuda", dtype=dtype)
        up = torch.randn(size, device="cuda", dtype=dtype)

        y_triton = swiglu_fused(gate, up)
        y_torch = swiglu_torch(gate, up)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("seq_len", [128, 512, 1024, 2048])
    @pytest.mark.parametrize("ffn_dim", [4096, 11008, 13824])
    def test_correctness_3d(self, batch_size: int, seq_len: int, ffn_dim: int):
        """Test SwiGLU on 3D input (batch, seq_len, ffn_dim)."""
        gate = torch.randn(batch_size, seq_len, ffn_dim, device="cuda", dtype=torch.float16)
        up = torch.randn_like(gate)

        y_triton = swiglu_fused(gate, up)
        y_torch = swiglu_torch(gate, up)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    def test_correctness_2d(self):
        """Test SwiGLU on 2D input (batch, ffn_dim)."""
        batch_size = 32
        ffn_dim = 11008

        gate = torch.randn(batch_size, ffn_dim, device="cuda", dtype=torch.float16)
        up = torch.randn_like(gate)

        y_triton = swiglu_fused(gate, up)
        y_torch = swiglu_torch(gate, up)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    def test_llama_7b_shapes(self):
        """Test with LLaMA 7B FFN dimensions."""
        # LLaMA 7B: hidden_dim=4096, ffn_dim=11008
        batch_size = 1
        seq_len = 2048
        ffn_dim = 11008

        gate = torch.randn(batch_size, seq_len, ffn_dim, device="cuda", dtype=torch.float16)
        up = torch.randn_like(gate)

        y_triton = swiglu_fused(gate, up)
        y_torch = swiglu_torch(gate, up)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    def test_llama_13b_shapes(self):
        """Test with LLaMA 13B FFN dimensions."""
        # LLaMA 13B: hidden_dim=5120, ffn_dim=13824
        batch_size = 1
        seq_len = 1024
        ffn_dim = 13824

        gate = torch.randn(batch_size, seq_len, ffn_dim, device="cuda", dtype=torch.float16)
        up = torch.randn_like(gate)

        y_triton = swiglu_fused(gate, up)
        y_torch = swiglu_torch(gate, up)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    def test_output_shape_preserved(self):
        """Verify output shape matches input shape."""
        shapes = [
            (4096,),
            (32, 11008),
            (2, 1024, 11008),
            (2, 4, 256, 4096),
        ]

        for shape in shapes:
            gate = torch.randn(shape, device="cuda", dtype=torch.float16)
            up = torch.randn(shape, device="cuda", dtype=torch.float16)
            y = swiglu_fused(gate, up)
            assert y.shape == gate.shape, f"Shape mismatch: {y.shape} vs {gate.shape}"

    def test_dtype_preserved(self):
        """Verify output dtype matches input dtype."""
        for dtype in [torch.float16, torch.float32]:
            gate = torch.randn(1024, device="cuda", dtype=dtype)
            up = torch.randn_like(gate)
            y = swiglu_fused(gate, up)
            assert y.dtype == dtype, f"Dtype mismatch: {y.dtype} vs {dtype}"

    def test_numerical_stability_large_values(self):
        """Test stability with large input values."""
        gate = torch.randn(1024, device="cuda", dtype=torch.float16) * 10
        up = torch.randn_like(gate) * 10

        y_triton = swiglu_fused(gate, up)
        y_torch = swiglu_torch(gate, up)

        assert not torch.isnan(y_triton).any(), "NaN in output"
        assert not torch.isinf(y_triton).any(), "Inf in output"
        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    def test_numerical_stability_small_values(self):
        """Test stability with small input values."""
        gate = torch.randn(1024, device="cuda", dtype=torch.float16) * 1e-4
        up = torch.randn_like(gate) * 1e-4

        y_triton = swiglu_fused(gate, up)
        y_torch = swiglu_torch(gate, up)

        assert not torch.isnan(y_triton).any(), "NaN in output"
        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    def test_zero_gate(self):
        """Test with zero gate (silu(0) = 0)."""
        gate = torch.zeros(1024, device="cuda", dtype=torch.float16)
        up = torch.randn(1024, device="cuda", dtype=torch.float16)

        y = swiglu_fused(gate, up)

        # silu(0) * up = 0 * up = 0
        assert y.abs().max() < 1e-6, "Output should be zero when gate is zero"

    def test_zero_up(self):
        """Test with zero up tensor."""
        gate = torch.randn(1024, device="cuda", dtype=torch.float16)
        up = torch.zeros(1024, device="cuda", dtype=torch.float16)

        y = swiglu_fused(gate, up)

        # silu(gate) * 0 = 0
        assert y.abs().max() < 1e-6, "Output should be zero when up is zero"

    def test_silu_identity(self):
        """Test that swiglu with up=1 gives silu(gate)."""
        gate = torch.randn(1024, device="cuda", dtype=torch.float16)
        up = torch.ones_like(gate)

        y_swiglu = swiglu_fused(gate, up)
        y_silu = F.silu(gate)

        torch.testing.assert_close(y_swiglu, y_silu, rtol=1e-2, atol=1e-3)

    def test_negative_values(self):
        """Test with predominantly negative values."""
        gate = -torch.rand(1024, device="cuda", dtype=torch.float16) * 5
        up = torch.randn_like(gate)

        y_triton = swiglu_fused(gate, up)
        y_torch = swiglu_torch(gate, up)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)


class TestSwiGLUModule:
    """Test suite for SwiGLU nn.Module."""

    def test_module_forward(self):
        """Test forward pass through module."""
        batch_size = 2
        seq_len = 1024
        ffn_dim = 11008

        module = SwiGLU()
        gate = torch.randn(batch_size, seq_len, ffn_dim, device="cuda", dtype=torch.float16)
        up = torch.randn_like(gate)

        y = module(gate, up)

        assert y.shape == gate.shape
        assert y.dtype == gate.dtype

    def test_module_correctness(self):
        """Test module correctness against reference."""
        batch_size = 4
        seq_len = 512
        ffn_dim = 4096

        module = SwiGLU()
        gate = torch.randn(batch_size, seq_len, ffn_dim, device="cuda", dtype=torch.float16)
        up = torch.randn_like(gate)

        y_module = module(gate, up)
        y_ref = swiglu_torch(gate, up)

        torch.testing.assert_close(y_module, y_ref, rtol=1e-2, atol=1e-3)


class TestAgainstPyTorchSiLU:
    """Test against torch.nn.functional.silu."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [512, 2048])
    @pytest.mark.parametrize("ffn_dim", [4096, 11008])
    def test_against_pytorch_silu(self, batch_size: int, seq_len: int, ffn_dim: int):
        """Verify swiglu_fused matches F.silu(gate) * up."""
        gate = torch.randn(batch_size, seq_len, ffn_dim, device="cuda", dtype=torch.float16)
        up = torch.randn_like(gate)

        y_triton = swiglu_fused(gate, up)
        y_pytorch = F.silu(gate) * up

        torch.testing.assert_close(y_triton, y_pytorch, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
