"""Fused MoE (Mixture-of-Experts) dispatch - cross-platform Triton kernel.

A single Triton-native forward pass for sparse expert routing: router scoring +
top-k gating, token permutation to expert-contiguous layout, a fused gate/up
expert GEMM with in-register SiLU, a grouped down projection, and a weighted
unpermute/combine. Because it is pure Triton it ships as a "universal" Kernel Hub
build - one artifact that runs on NVIDIA (SM80/SM90) and AMD (MI300X).

Loading from the Hub::

    from kernels import get_kernel

    moe = get_kernel("bassrehab/moe-dispatch", version=0, trust_remote_code=True)
    out, idx, weights = moe.fused_moe_forward(
        hidden_states, router_weight, w_gate, w_up, w_down,
        num_experts=8, top_k=2, gating="softmax",
    )

``trust_remote_code=True`` is required until the publisher is added to the Kernel
Hub trusted-publisher list.
"""

from .router import moe_router
from .permute import permute_tokens, unpermute_tokens
from .expert_gemm import grouped_gemm, expert_ffn_triton
from .fused_moe import fused_moe_forward, fused_expert_ffn

# subhadipmitra, 2026-07-19: __all__ IS the versioned public API on the Kernel
# Hub - the versioning guarantee only covers symbols listed here; anything else
# is private and may change without a version bump. Keep this list small and
# deliberate so the v-branch API surface stays stable.
__all__ = [
    "fused_moe_forward",
    "fused_expert_ffn",
    "moe_router",
    "permute_tokens",
    "unpermute_tokens",
    "grouped_gemm",
    "expert_ffn_triton",
]
