"""
Tests for RMSNorm Triton kernel.

Validates numerical correctness against PyTorch reference implementation.
Tests various input shapes, dtypes, and edge cases.
"""

import pytest
import torch

from triton_kernels.rmsnorm import (
    rmsnorm,
    rmsnorm_torch,
    rmsnorm_residual_fused,
    rmsnorm_residual_torch,
    TritonRMSNorm,
)


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestRMSNorm:
    """Test suite for RMSNorm kernel."""

    @pytest.mark.parametrize("hidden_dim", [64, 128, 256, 512, 1024, 2048, 4096, 8192])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    def test_correctness_1d(self, hidden_dim: int, dtype: torch.dtype):
        """Test RMSNorm on 1D input (single vector)."""
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("BF16 not supported on this GPU")

        x = torch.randn(hidden_dim, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_dim, device="cuda", dtype=dtype)
        eps = 1e-6

        # Triton kernel
        y_triton = rmsnorm(x, weight, eps)

        # PyTorch reference
        y_torch = rmsnorm_torch(x, weight, eps)

        # Check correctness
        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("seq_len", [128, 512, 1024, 2048])
    @pytest.mark.parametrize("hidden_dim", [1024, 2048, 4096])
    def test_correctness_3d(self, batch_size: int, seq_len: int, hidden_dim: int):
        """Test RMSNorm on 3D input (batch, seq_len, hidden_dim)."""
        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        eps = 1e-6

        y_triton = rmsnorm(x, weight, eps)
        y_torch = rmsnorm_torch(x, weight, eps)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    @pytest.mark.parametrize("hidden_dim", [1024, 4096])
    def test_correctness_2d(self, hidden_dim: int):
        """Test RMSNorm on 2D input (batch, hidden_dim)."""
        batch_size = 32
        x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float16)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        eps = 1e-6

        y_triton = rmsnorm(x, weight, eps)
        y_torch = rmsnorm_torch(x, weight, eps)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    def test_correctness_4d(self):
        """Test RMSNorm on 4D input."""
        x = torch.randn(2, 4, 128, 1024, device="cuda", dtype=torch.float16)
        weight = torch.randn(1024, device="cuda", dtype=torch.float16)
        eps = 1e-6

        y_triton = rmsnorm(x, weight, eps)
        y_torch = rmsnorm_torch(x, weight, eps)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    def test_output_shape_preserved(self):
        """Verify output shape matches input shape."""
        shapes = [
            (4096,),
            (32, 4096),
            (2, 1024, 4096),
            (2, 4, 256, 4096),
        ]
        hidden_dim = 4096
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)

        for shape in shapes:
            x = torch.randn(shape, device="cuda", dtype=torch.float16)
            y = rmsnorm(x, weight)
            assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

    def test_dtype_preserved(self):
        """Verify output dtype matches input dtype."""
        hidden_dim = 1024

        for dtype in [torch.float16, torch.float32]:
            x = torch.randn(32, hidden_dim, device="cuda", dtype=dtype)
            weight = torch.randn(hidden_dim, device="cuda", dtype=dtype)
            y = rmsnorm(x, weight)
            assert y.dtype == dtype, f"Dtype mismatch: {y.dtype} vs {dtype}"

    def test_numerical_stability_small_values(self):
        """Test stability with very small input values."""
        hidden_dim = 1024
        x = torch.randn(32, hidden_dim, device="cuda", dtype=torch.float16) * 1e-4
        weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float16)

        y_triton = rmsnorm(x, weight, eps=1e-6)
        y_torch = rmsnorm_torch(x, weight, eps=1e-6)

        # Should not have NaN or Inf
        assert not torch.isnan(y_triton).any(), "NaN in output"
        assert not torch.isinf(y_triton).any(), "Inf in output"
        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    def test_numerical_stability_large_values(self):
        """Test stability with large input values."""
        hidden_dim = 1024
        x = torch.randn(32, hidden_dim, device="cuda", dtype=torch.float16) * 1e3
        weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float16)

        y_triton = rmsnorm(x, weight, eps=1e-6)
        y_torch = rmsnorm_torch(x, weight, eps=1e-6)

        assert not torch.isnan(y_triton).any(), "NaN in output"
        assert not torch.isinf(y_triton).any(), "Inf in output"
        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    def test_zero_input(self):
        """Test behavior with zero input."""
        hidden_dim = 1024
        x = torch.zeros(32, hidden_dim, device="cuda", dtype=torch.float16)
        weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float16)

        y = rmsnorm(x, weight, eps=1e-6)

        # With zero input, output should be zero (0 * anything = 0)
        assert not torch.isnan(y).any(), "NaN in output for zero input"
        # Output should be near zero
        assert y.abs().max() < 1e-3, "Output should be near zero for zero input"

    @pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-8])
    def test_different_eps(self, eps: float):
        """Test with different epsilon values."""
        hidden_dim = 1024
        x = torch.randn(32, hidden_dim, device="cuda", dtype=torch.float16)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)

        y_triton = rmsnorm(x, weight, eps)
        y_torch = rmsnorm_torch(x, weight, eps)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)


class TestTritonRMSNormModule:
    """Test suite for TritonRMSNorm nn.Module."""

    def test_module_forward(self):
        """Test forward pass through module."""
        hidden_size = 4096
        batch_size = 2
        seq_len = 1024

        module = TritonRMSNorm(hidden_size).cuda().half()
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)

        y = module(x)

        assert y.shape == x.shape
        assert y.dtype == x.dtype

    def test_module_parameters(self):
        """Test that module has correct parameters."""
        hidden_size = 4096
        module = TritonRMSNorm(hidden_size)

        params = list(module.parameters())
        assert len(params) == 1
        assert params[0].shape == (hidden_size,)

    def test_module_correctness(self):
        """Test module correctness against reference."""
        hidden_size = 2048
        batch_size = 4
        seq_len = 512

        module = TritonRMSNorm(hidden_size, eps=1e-6).cuda()
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float32)

        y_module = module(x)
        y_ref = rmsnorm_torch(x, module.weight, eps=1e-6)

        torch.testing.assert_close(y_module, y_ref, rtol=1e-2, atol=1e-3)


class TestAgainstTorchRMSNorm:
    """Test against torch.nn.RMSNorm if available (PyTorch 2.4+)."""

    @pytest.fixture
    def check_torch_rmsnorm(self):
        """Check if torch.nn.RMSNorm is available."""
        if not hasattr(torch.nn, "RMSNorm"):
            pytest.skip("torch.nn.RMSNorm not available (requires PyTorch 2.4+)")

    @pytest.mark.parametrize("hidden_dim", [1024, 2048, 4096])
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [512, 2048])
    def test_against_torch_rmsnorm(
        self, check_torch_rmsnorm, hidden_dim: int, batch_size: int, seq_len: int
    ):
        """Compare Triton RMSNorm against PyTorch's native implementation."""
        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)

        # PyTorch native
        torch_norm = torch.nn.RMSNorm(hidden_dim, eps=1e-6).cuda().half()

        # Triton
        triton_norm = TritonRMSNorm(hidden_dim, eps=1e-6).cuda().half()

        # Copy weights
        triton_norm.weight.data.copy_(torch_norm.weight.data)

        y_torch = torch_norm(x)
        y_triton = triton_norm(x)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)


class TestRMSNormResidualFused:
    """Test suite for fused RMSNorm + residual kernel."""

    @pytest.mark.parametrize("hidden_dim", [64, 128, 256, 512, 1024, 2048, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_correctness_1d(self, hidden_dim: int, dtype: torch.dtype):
        """Test fused RMSNorm + residual on 1D input."""
        x = torch.randn(hidden_dim, device="cuda", dtype=dtype)
        residual = torch.randn(hidden_dim, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_dim, device="cuda", dtype=dtype)
        eps = 1e-6

        y_triton = rmsnorm_residual_fused(x, residual, weight, eps)
        y_torch = rmsnorm_residual_torch(x, residual, weight, eps)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [128, 512, 1024, 2048])
    @pytest.mark.parametrize("hidden_dim", [1024, 2048, 4096])
    def test_correctness_3d(self, batch_size: int, seq_len: int, hidden_dim: int):
        """Test fused RMSNorm + residual on 3D input (batch, seq_len, hidden_dim)."""
        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
        residual = torch.randn_like(x)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        eps = 1e-6

        y_triton = rmsnorm_residual_fused(x, residual, weight, eps)
        y_torch = rmsnorm_residual_torch(x, residual, weight, eps)

        torch.testing.assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-3)

    def test_output_shape_preserved(self):
        """Verify output shape matches input shape for fused variant."""
        shapes = [
            (4096,),
            (32, 4096),
            (2, 1024, 4096),
        ]
        hidden_dim = 4096
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)

        for shape in shapes:
            x = torch.randn(shape, device="cuda", dtype=torch.float16)
            residual = torch.randn(shape, device="cuda", dtype=torch.float16)
            y = rmsnorm_residual_fused(x, residual, weight)
            assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

    def test_equivalence_to_separate_ops(self):
        """Verify fused kernel gives same result as separate add + rmsnorm."""
        hidden_dim = 2048
        batch_size = 4
        seq_len = 512

        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
        residual = torch.randn_like(x)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        eps = 1e-6

        # Fused
        y_fused = rmsnorm_residual_fused(x, residual, weight, eps)

        # Separate operations
        y_separate = rmsnorm(x + residual, weight, eps)

        torch.testing.assert_close(y_fused, y_separate, rtol=1e-2, atol=1e-3)

    def test_numerical_stability(self):
        """Test numerical stability with mixed scales."""
        hidden_dim = 2048
        batch_size = 4
        seq_len = 256

        # x is small, residual is large
        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16) * 1e-3
        residual = torch.randn_like(x) * 1e2
        weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float16)

        y = rmsnorm_residual_fused(x, residual, weight, eps=1e-6)

        assert not torch.isnan(y).any(), "NaN in output"
        assert not torch.isinf(y).any(), "Inf in output"

    def test_zero_residual(self):
        """Test with zero residual (should equal plain rmsnorm)."""
        hidden_dim = 1024
        batch_size = 8
        seq_len = 256

        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
        residual = torch.zeros_like(x)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        eps = 1e-6

        y_fused = rmsnorm_residual_fused(x, residual, weight, eps)
        y_plain = rmsnorm(x, weight, eps)

        torch.testing.assert_close(y_fused, y_plain, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
