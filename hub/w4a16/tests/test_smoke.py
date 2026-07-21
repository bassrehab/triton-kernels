"""Smoke test for the universal w4a16 kernel.

Quantizes a weight, runs a small w4a16_gemm forward, and checks shape + finiteness.
Triton JIT-compiles at runtime, so this needs a CUDA/ROCm GPU; it is skipped
automatically on CPU-only hosts and is meant to run on the build/benchmark box.
"""
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a GPU"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_kernel():
    """Load the way a user would, falling back to the torch-ext source tree.

    Once build/torch-universal exists, get_local_kernel exercises the packaged
    layout; before that we import straight from torch-ext so the test is useful
    during development.
    """
    try:
        from kernels import get_local_kernel

        return get_local_kernel(PROJECT_ROOT, "w4a16")
    except Exception:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "torch-ext"))
        import w4a16

        return w4a16


def test_w4a16_gemm_smoke():
    torch.manual_seed(0)
    k = _load_kernel()

    K, N, group_size, M = 2048, 512, 128, 8
    packed, scales, zeros = k.quantize_weight_int4_grouped(torch.randn(K, N) * 0.1, group_size)

    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    y = k.w4a16_gemm(x, packed.cuda(), scales.cuda().half(), zeros.cuda().half(), group_size)

    assert y.shape == (M, N)
    assert torch.isfinite(y).all()
