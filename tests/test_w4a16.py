"""Correctness tests for the W4A16 GEMM kernel.

Two layers:
  - `TestW4A16Packing`: pure-PyTorch weight-prep utilities (pack/unpack, quant
    format). These run on CPU, no GPU or Triton needed.
  - `TestW4A16Kernel`: the Triton `w4a16_gemm` kernel vs the PyTorch reference.
    Skipped without a GPU (Triton JIT-compiles at runtime).
"""
import pytest
import torch

from reference.w4a16_reference import (
    quantize_weight_int4_grouped,
    pack_int4,
    unpack_int4,
    dequantize_int4,
    w4a16_reference,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton W4A16 kernel requires a GPU"
)


class TestW4A16Packing:
    """Pure-PyTorch weight-prep tests (CPU)."""

    def test_pack_unpack_roundtrip(self):
        torch.manual_seed(0)
        q = torch.randint(0, 16, (256, 128), dtype=torch.int32)
        assert torch.equal(unpack_int4(pack_int4(q), 256), q)

    @pytest.mark.parametrize("symmetric", [False, True])
    def test_quantize_format(self, symmetric):
        torch.manual_seed(0)
        K, N, G = 512, 256, 128
        packed, scales, zeros = quantize_weight_int4_grouped(torch.randn(K, N), G, symmetric)
        assert packed.shape == (K // 8, N) and packed.dtype == torch.int32
        assert scales.shape == (K // G, N) and zeros.shape == (K // G, N)
        q = unpack_int4(packed, K)
        assert int(q.min()) >= 0 and int(q.max()) <= 15

    @pytest.mark.parametrize("symmetric", [False, True])
    def test_dequant_within_quant_error(self, symmetric):
        torch.manual_seed(0)
        K, N, G = 512, 256, 128
        W = torch.randn(K, N)
        packed, scales, zeros = quantize_weight_int4_grouped(W, G, symmetric)
        Wdq = dequantize_int4(packed, scales, zeros, G, K)
        # 4-bit grouped quant of Gaussian weights: a few percent mean relative error.
        rel = (Wdq - W).abs().mean() / W.abs().mean()
        assert rel < 0.15


@requires_cuda
class TestW4A16Kernel:
    """Triton kernel vs PyTorch reference (GPU)."""

    @pytest.mark.parametrize("M,K,N", [
        (1, 256, 512),       # decode: single token (split-K path, SPLIT_K=1)
        (1, 4096, 512),      # decode: split-K path with SPLIT_K=8
        (8, 2048, 256),      # small batch: split-K path
        (16, 512, 256),      # small batch: non-split path
        (64, 1024, 512),
        (128, 4096, 4096),   # Llama-8B attention projection shape
    ])
    @pytest.mark.parametrize("group_size", [64, 128])
    @pytest.mark.parametrize("symmetric", [False, True])
    def test_matches_reference(self, M, K, N, group_size, symmetric):
        from triton_kernels.w4a16 import w4a16_gemm

        torch.manual_seed(0)
        dev = "cuda"
        x = torch.randn(M, K, device=dev, dtype=torch.float16)
        packed, scales, zeros = quantize_weight_int4_grouped(
            torch.randn(K, N) * 0.1, group_size, symmetric
        )
        packed = packed.to(dev)
        scales = scales.to(dev).half()
        zeros = zeros.to(dev).half()

        y_ref = w4a16_reference(x, packed, scales, zeros, group_size)
        y_kernel = w4a16_gemm(x, packed, scales, zeros, group_size)

        # Both dequantize the same int4 values; difference is only FP reordering.
        torch.testing.assert_close(y_kernel, y_ref, rtol=1e-2, atol=1e-2)

    def test_output_shape_and_finite(self):
        from triton_kernels.w4a16 import w4a16_gemm

        torch.manual_seed(0)
        M, K, N, G = 32, 1024, 512, 128
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        packed, scales, zeros = quantize_weight_int4_grouped(torch.randn(K, N) * 0.1, G, False)
        y = w4a16_gemm(x, packed.cuda(), scales.cuda().half(), zeros.cuda().half(), G)
        assert y.shape == (M, N)
        assert torch.isfinite(y).all()
