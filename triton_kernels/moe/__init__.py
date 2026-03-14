"""
Fused MoE (Mixture-of-Experts) dispatch kernels implemented in Triton.

This submodule provides high-performance Triton kernels for the complete MoE
forward pass: router scoring, top-k gating, token permutation, expert GEMMs,
token unpermutation, and weighted combination.

The kernels use only Triton primitives for cross-platform portability
(NVIDIA + AMD via Triton backends).
"""

from triton_kernels.moe.router import moe_router, moe_router_torch

__all__ = [
    "moe_router",
    "moe_router_torch",
]
