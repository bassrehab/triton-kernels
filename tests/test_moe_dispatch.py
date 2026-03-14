"""
Tests for MoE dispatch Triton kernels.

Phase 1: Router kernel and reference implementation correctness.
Phase 2: Permute/unpermute kernel correctness and roundtrip validation.
"""

import pytest
import torch

from triton_kernels.moe.router import moe_router, moe_router_torch
from triton_kernels.moe.permute import (
    permute_tokens,
    unpermute_tokens,
    permute_tokens_torch,
    unpermute_tokens_torch,
    compute_permutation_indices,
)


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Model configurations from real MoE deployments
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "mixtral-8x7b": {"num_experts": 8, "top_k": 2, "hidden_dim": 4096},
    "mixtral-8x22b": {"num_experts": 8, "top_k": 2, "hidden_dim": 6144},
    "deepseek-v3": {"num_experts": 256, "top_k": 8, "hidden_dim": 7168},
    "qwen2-moe-57b": {"num_experts": 64, "top_k": 4, "hidden_dim": 3584},
}


class TestMoERouterSoftmax:
    """Test MoE router with softmax gating against PyTorch reference."""

    @pytest.mark.parametrize("num_tokens", [1, 32, 128, 512])
    @pytest.mark.parametrize("num_experts,top_k", [(8, 2), (64, 4), (256, 8)])
    def test_topk_indices_valid(self, num_tokens: int, num_experts: int, top_k: int):
        """Selected expert indices must be in [0, num_experts) and unique per token."""
        hidden_dim = 256
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float16)

        indices, weights, logits = moe_router(hidden_states, router_weight, top_k, gating="softmax")

        assert indices.shape == (num_tokens, top_k)
        assert weights.shape == (num_tokens, top_k)
        assert (indices >= 0).all() and (indices < num_experts).all(), "Indices out of range"

        # Check uniqueness per token
        for t in range(num_tokens):
            assert len(indices[t].unique()) == top_k, f"Duplicate experts for token {t}"

    @pytest.mark.parametrize("num_tokens", [1, 32, 128, 512])
    @pytest.mark.parametrize("num_experts,top_k", [(8, 2), (64, 4)])
    def test_softmax_weights_match_reference(self, num_tokens: int, num_experts: int, top_k: int):
        """Triton router softmax weights must match PyTorch reference."""
        hidden_dim = 256
        # Use a fixed seed for reproducibility in tie-breaking
        torch.manual_seed(42)
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float16)

        indices_triton, weights_triton, _ = moe_router(
            hidden_states, router_weight, top_k, gating="softmax",
        )
        indices_torch, weights_torch, _ = moe_router_torch(
            hidden_states, router_weight, top_k, gating="softmax",
        )

        # Compare expert selections — allow mismatches when scores are near-tied
        # (FP rounding can cause Triton vs PyTorch to pick different experts)
        match_count = 0
        for t in range(num_tokens):
            triton_sorted = indices_triton[t].sort().values
            torch_sorted = indices_torch[t].sort().values
            if torch.equal(triton_sorted, torch_sorted):
                match_count += 1

        # At least 90% of tokens should agree on expert selection
        match_ratio = match_count / num_tokens
        assert match_ratio >= 0.9, (
            f"Only {match_ratio:.1%} of tokens agree on expert selection "
            f"(expected >= 90%)"
        )

        # For tokens that agree, weights should be close
        torch.testing.assert_close(
            weights_triton.float().sort(dim=-1).values,
            weights_torch.float().sort(dim=-1).values,
            rtol=1e-2, atol=5e-3,
        )

    @pytest.mark.parametrize("num_tokens", [1, 64, 256])
    def test_softmax_weights_sum_to_topk_fraction(self, num_tokens: int):
        """Softmax top-k weights should sum to a value <= 1.0 per token."""
        num_experts = 8
        top_k = 2
        hidden_dim = 128
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float16)

        _, weights, _ = moe_router(hidden_states, router_weight, top_k, gating="softmax")

        weight_sums = weights.float().sum(dim=-1)
        assert (weight_sums <= 1.0 + 1e-3).all(), f"Weight sums exceed 1.0: {weight_sums}"
        assert (weight_sums > 0.0).all(), "Weight sums should be positive"

    def test_softmax_weights_non_negative(self):
        """All softmax gating weights must be non-negative."""
        num_tokens, num_experts, top_k, hidden_dim = 128, 8, 2, 256
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float16)

        _, weights, _ = moe_router(hidden_states, router_weight, top_k, gating="softmax")

        assert (weights >= 0).all(), "Softmax weights must be non-negative"
        # Top-1 weight should always be positive
        assert (weights[:, 0] > 0).all(), "Top-1 weight must be positive"


class TestMoERouterSigmoid:
    """Test MoE router with sigmoid gating."""

    @pytest.mark.parametrize("num_tokens", [1, 32, 128])
    @pytest.mark.parametrize("num_experts,top_k", [(8, 2), (64, 4)])
    def test_sigmoid_weights_match_reference(self, num_tokens: int, num_experts: int, top_k: int):
        """Triton sigmoid router must match PyTorch reference."""
        hidden_dim = 256
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float16)

        indices_triton, weights_triton, _ = moe_router(
            hidden_states, router_weight, top_k, gating="sigmoid",
        )
        indices_torch, weights_torch, _ = moe_router_torch(
            hidden_states, router_weight, top_k, gating="sigmoid",
        )

        # Same experts selected
        for t in range(num_tokens):
            triton_sorted = indices_triton[t].sort().values
            torch_sorted = indices_torch[t].sort().values
            assert torch.equal(triton_sorted, torch_sorted), (
                f"Token {t}: Triton {indices_triton[t].tolist()} vs PyTorch {indices_torch[t].tolist()}"
            )

        # Weights close
        torch.testing.assert_close(
            weights_triton.float().sort(dim=-1).values,
            weights_torch.float().sort(dim=-1).values,
            rtol=1e-2, atol=1e-3,
        )

    @pytest.mark.parametrize("num_tokens", [1, 64])
    def test_sigmoid_weights_normalized(self, num_tokens: int):
        """Sigmoid top-k weights should be normalized to sum to ~1.0."""
        num_experts, top_k, hidden_dim = 8, 2, 128
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float16)

        _, weights, _ = moe_router(hidden_states, router_weight, top_k, gating="sigmoid")

        weight_sums = weights.float().sum(dim=-1)
        torch.testing.assert_close(
            weight_sums, torch.ones_like(weight_sums), rtol=1e-2, atol=1e-2,
        )


class TestMoERouterModelConfigs:
    """Test router with realistic model configurations."""

    @pytest.mark.parametrize("model_name", ["mixtral-8x7b", "mixtral-8x22b"])
    @pytest.mark.parametrize("num_tokens", [1, 128, 1024])
    def test_small_expert_count(self, model_name: str, num_tokens: int):
        """Test with Mixtral-scale configs (8 experts)."""
        cfg = MODEL_CONFIGS[model_name]
        hidden_states = torch.randn(
            num_tokens, cfg["hidden_dim"], device="cuda", dtype=torch.float16,
        )
        router_weight = torch.randn(
            cfg["num_experts"], cfg["hidden_dim"], device="cuda", dtype=torch.float16,
        )

        indices, weights, logits = moe_router(
            hidden_states, router_weight, cfg["top_k"], gating="softmax",
        )

        assert indices.shape == (num_tokens, cfg["top_k"])
        assert weights.shape == (num_tokens, cfg["top_k"])
        assert logits.shape == (num_tokens, cfg["num_experts"])
        assert (indices >= 0).all() and (indices < cfg["num_experts"]).all()

    @pytest.mark.parametrize("model_name", ["qwen2-moe-57b"])
    @pytest.mark.parametrize("num_tokens", [1, 128])
    def test_medium_expert_count(self, model_name: str, num_tokens: int):
        """Test with Qwen2-scale configs (64 experts)."""
        cfg = MODEL_CONFIGS[model_name]
        hidden_states = torch.randn(
            num_tokens, cfg["hidden_dim"], device="cuda", dtype=torch.float16,
        )
        router_weight = torch.randn(
            cfg["num_experts"], cfg["hidden_dim"], device="cuda", dtype=torch.float16,
        )

        indices, weights, logits = moe_router(
            hidden_states, router_weight, cfg["top_k"], gating="softmax",
        )

        assert indices.shape == (num_tokens, cfg["top_k"])
        assert (indices >= 0).all() and (indices < cfg["num_experts"]).all()

    @pytest.mark.parametrize("num_tokens", [1, 32])
    def test_large_expert_count(self, num_tokens: int):
        """Test with DeepSeek-V3 scale (256 experts, top-8)."""
        cfg = MODEL_CONFIGS["deepseek-v3"]
        hidden_states = torch.randn(
            num_tokens, cfg["hidden_dim"], device="cuda", dtype=torch.float16,
        )
        router_weight = torch.randn(
            cfg["num_experts"], cfg["hidden_dim"], device="cuda", dtype=torch.float16,
        )

        indices, weights, logits = moe_router(
            hidden_states, router_weight, cfg["top_k"], gating="softmax",
        )

        assert indices.shape == (num_tokens, cfg["top_k"])
        assert (indices >= 0).all() and (indices < cfg["num_experts"]).all()

        # Check all 8 selected experts are unique per token
        for t in range(num_tokens):
            assert len(indices[t].unique()) == cfg["top_k"]


class TestMoERouterEdgeCases:
    """Edge cases and numerical stability tests."""

    def test_single_token(self):
        """Router works with a single token."""
        hidden_states = torch.randn(1, 256, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(8, 256, device="cuda", dtype=torch.float16)

        indices, weights, logits = moe_router(hidden_states, router_weight, top_k=2)

        assert indices.shape == (1, 2)
        assert not torch.isnan(weights).any()

    def test_topk_equals_num_experts(self):
        """When top_k == num_experts, all experts should be selected."""
        num_tokens, num_experts, hidden_dim = 16, 4, 128
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float16)

        indices, weights, _ = moe_router(hidden_states, router_weight, top_k=num_experts)

        for t in range(num_tokens):
            assert set(indices[t].tolist()) == set(range(num_experts))

        # Weights should sum to ~1.0 (full softmax)
        torch.testing.assert_close(
            weights.float().sum(dim=-1),
            torch.ones(num_tokens, device="cuda"),
            rtol=1e-2, atol=1e-2,
        )

    def test_no_nan_with_large_logits(self):
        """Numerical stability: large logits should not produce NaN."""
        num_tokens, num_experts, hidden_dim = 32, 8, 128
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16) * 100
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float16)

        indices, weights, logits = moe_router(hidden_states, router_weight, top_k=2)

        assert not torch.isnan(weights).any(), "NaN in weights with large logits"
        assert not torch.isinf(weights).any(), "Inf in weights with large logits"

    def test_no_nan_with_small_logits(self):
        """Numerical stability: very small logits should not produce NaN."""
        num_tokens, num_experts, hidden_dim = 32, 8, 128
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16) * 1e-4
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float16) * 1e-4

        indices, weights, logits = moe_router(hidden_states, router_weight, top_k=2)

        assert not torch.isnan(weights).any(), "NaN in weights with small logits"

    def test_output_dtypes(self):
        """Verify output tensor dtypes."""
        hidden_states = torch.randn(16, 128, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(8, 128, device="cuda", dtype=torch.float16)

        indices, weights, logits = moe_router(hidden_states, router_weight, top_k=2)

        assert indices.dtype == torch.int64
        assert weights.dtype == torch.float16
        assert logits.dtype == torch.float16

    def test_deterministic(self):
        """Same inputs should produce same outputs."""
        hidden_states = torch.randn(64, 256, device="cuda", dtype=torch.float16)
        router_weight = torch.randn(8, 256, device="cuda", dtype=torch.float16)

        idx1, w1, _ = moe_router(hidden_states, router_weight, top_k=2)
        idx2, w2, _ = moe_router(hidden_states, router_weight, top_k=2)

        assert torch.equal(idx1, idx2)
        assert torch.equal(w1, w2)


class TestMoEReferenceImplementation:
    """Test the full reference MoE implementation."""

    def test_reference_forward_shape(self):
        """Reference MoE produces correct output shape."""
        from reference.moe_reference import MoEReference

        num_tokens, hidden_dim, ffn_dim = 32, 256, 512
        num_experts, top_k = 8, 2

        moe = MoEReference(hidden_dim, ffn_dim, num_experts, top_k).cuda().half()
        x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)

        output, routing = moe(x)

        assert output.shape == (num_tokens, hidden_dim)
        assert routing.top_k_indices.shape == (num_tokens, top_k)
        assert routing.top_k_weights.shape == (num_tokens, top_k)

    def test_reference_forward_no_nan(self):
        """Reference MoE should not produce NaN."""
        from reference.moe_reference import MoEReference

        moe = MoEReference(256, 512, 8, 2).cuda().half()
        x = torch.randn(64, 256, device="cuda", dtype=torch.float16)

        output, _ = moe(x)

        assert not torch.isnan(output).any(), "NaN in reference MoE output"
        assert not torch.isinf(output).any(), "Inf in reference MoE output"

    def test_reference_permute_unpermute_roundtrip(self):
        """Permute -> unpermute should recover weighted original."""
        from reference.moe_reference import permute_tokens, unpermute_tokens

        num_tokens, hidden_dim, num_experts, top_k = 64, 128, 8, 2
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)

        # Random routing
        top_k_indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")
        top_k_weights = torch.softmax(
            torch.randn(num_tokens, top_k, device="cuda", dtype=torch.float32), dim=-1,
        ).half()

        # Permute
        permuted, offsets, restore = permute_tokens(hidden_states, top_k_indices, num_experts)

        # Use identity transform (no expert FFN)
        # Unpermute with weights
        output = unpermute_tokens(permuted, restore, top_k_weights)

        # Output should equal weighted sum of duplicated inputs
        expected = (
            hidden_states.unsqueeze(1).expand(-1, top_k, -1)
            * top_k_weights.unsqueeze(-1)
        ).sum(dim=1)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-3)

    def test_reference_sigmoid_gating(self):
        """Reference MoE works with sigmoid gating."""
        from reference.moe_reference import MoEReference

        moe = MoEReference(256, 512, 8, 2, gating="sigmoid").cuda().half()
        x = torch.randn(32, 256, device="cuda", dtype=torch.float16)

        output, routing = moe(x)

        assert output.shape == (32, 256)
        assert not torch.isnan(output).any()

        # Sigmoid weights should be normalized
        weight_sums = routing.top_k_weights.float().sum(dim=-1)
        torch.testing.assert_close(
            weight_sums, torch.ones_like(weight_sums), rtol=1e-2, atol=1e-2,
        )


# ---------------------------------------------------------------------------
# Phase 2: Permute/Unpermute Tests
# ---------------------------------------------------------------------------


class TestPermuteTokens:
    """Test token permutation Triton kernel against PyTorch reference."""

    @pytest.mark.parametrize("num_tokens", [1, 32, 128, 512])
    @pytest.mark.parametrize("num_experts,top_k", [(8, 2), (64, 4)])
    @pytest.mark.parametrize("hidden_dim", [128, 1024, 4096])
    def test_permute_matches_reference(
        self, num_tokens: int, num_experts: int, top_k: int, hidden_dim: int,
    ):
        """Triton permute output must match PyTorch reference exactly."""
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        top_k_indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")

        permuted_triton, offsets_triton, _, _ = permute_tokens(
            hidden_states, top_k_indices, num_experts,
        )
        permuted_torch, offsets_torch, _, _ = permute_tokens_torch(
            hidden_states, top_k_indices, num_experts,
        )

        torch.testing.assert_close(permuted_triton, permuted_torch, rtol=0, atol=0)
        assert torch.equal(offsets_triton, offsets_torch)

    @pytest.mark.parametrize("num_tokens", [1, 64, 256])
    def test_permute_output_shape(self, num_tokens: int):
        """Permuted output has correct shape."""
        num_experts, top_k, hidden_dim = 8, 2, 512
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        top_k_indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")

        permuted, offsets, sorted_idx, restore_idx = permute_tokens(
            hidden_states, top_k_indices, num_experts,
        )

        assert permuted.shape == (num_tokens * top_k, hidden_dim)
        assert offsets.shape == (num_experts + 1,)
        assert sorted_idx.shape == (num_tokens * top_k,)
        assert restore_idx.shape == (num_tokens * top_k,)

    def test_expert_offsets_sum(self):
        """Expert offsets should span all tokens * top_k."""
        num_tokens, num_experts, top_k, hidden_dim = 64, 8, 2, 256
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        top_k_indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")

        _, offsets, _, _ = permute_tokens(hidden_states, top_k_indices, num_experts)

        assert offsets[0].item() == 0
        assert offsets[-1].item() == num_tokens * top_k
        # Offsets should be non-decreasing
        assert (offsets[1:] >= offsets[:-1]).all()

    def test_expert_contiguity(self):
        """Tokens for each expert should be contiguous in permuted output."""
        num_tokens, num_experts, top_k, hidden_dim = 128, 8, 2, 64
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        top_k_indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")

        _, offsets, sorted_idx, _ = permute_tokens(hidden_states, top_k_indices, num_experts)

        flat_experts = top_k_indices.reshape(-1)
        sorted_experts = flat_experts[sorted_idx]

        for e in range(num_experts):
            start = offsets[e].item()
            end = offsets[e + 1].item()
            if start < end:
                assert (sorted_experts[start:end] == e).all(), (
                    f"Expert {e} tokens not contiguous"
                )

    def test_permute_preserves_values(self):
        """Permutation should not alter token values, only their positions."""
        num_tokens, hidden_dim, num_experts, top_k = 64, 256, 8, 2
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        top_k_indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")

        permuted, _, sorted_idx, _ = permute_tokens(hidden_states, top_k_indices, num_experts)

        # Each permuted row should exactly match the source token
        expanded = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)
        expected = expanded[sorted_idx]

        torch.testing.assert_close(permuted, expected, rtol=0, atol=0)


class TestUnpermuteTokens:
    """Test token unpermutation Triton kernel against PyTorch reference."""

    @pytest.mark.parametrize("num_tokens", [1, 32, 128, 512])
    @pytest.mark.parametrize("num_experts,top_k", [(8, 2), (64, 4)])
    @pytest.mark.parametrize("hidden_dim", [128, 1024, 4096])
    def test_unpermute_matches_reference(
        self, num_tokens: int, num_experts: int, top_k: int, hidden_dim: int,
    ):
        """Triton unpermute must match PyTorch reference."""
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        top_k_indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")
        top_k_weights = torch.softmax(
            torch.randn(num_tokens, top_k, device="cuda", dtype=torch.float32), dim=-1,
        ).half()

        # Permute (use torch reference to get shared sorted/restore indices)
        permuted, _, sorted_idx, restore_idx = permute_tokens_torch(
            hidden_states, top_k_indices, num_experts,
        )

        # Simulate expert outputs (identity = use permuted tokens as-is)
        expert_outputs = permuted.clone()

        out_triton = unpermute_tokens(expert_outputs, top_k_weights, restore_idx, num_tokens, top_k)
        out_torch = unpermute_tokens_torch(expert_outputs, top_k_weights, restore_idx, num_tokens, top_k)

        torch.testing.assert_close(out_triton, out_torch, rtol=1e-2, atol=1e-3)

    def test_unpermute_output_shape(self):
        """Unpermuted output has correct shape."""
        num_tokens, num_experts, top_k, hidden_dim = 64, 8, 2, 512
        expert_outputs = torch.randn(num_tokens * top_k, hidden_dim, device="cuda", dtype=torch.float16)
        top_k_weights = torch.ones(num_tokens, top_k, device="cuda", dtype=torch.float16) / top_k
        restore_idx = torch.arange(num_tokens * top_k, device="cuda")

        output = unpermute_tokens(expert_outputs, top_k_weights, restore_idx, num_tokens, top_k)

        assert output.shape == (num_tokens, hidden_dim)


class TestPermuteUnpermuteRoundtrip:
    """Test full permute -> unpermute roundtrip with Triton kernels."""

    @pytest.mark.parametrize("num_tokens", [1, 32, 128, 512])
    @pytest.mark.parametrize("num_experts,top_k", [(8, 2), (64, 4), (256, 8)])
    def test_roundtrip_identity(self, num_tokens: int, num_experts: int, top_k: int):
        """Permute -> identity transform -> unpermute should equal weighted input."""
        hidden_dim = 256
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        top_k_indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")
        top_k_weights = torch.softmax(
            torch.randn(num_tokens, top_k, device="cuda", dtype=torch.float32), dim=-1,
        ).half()

        # Permute
        permuted, offsets, sorted_idx, restore_idx = permute_tokens(
            hidden_states, top_k_indices, num_experts,
        )

        # Identity expert (no transformation)
        # Unpermute
        output = unpermute_tokens(permuted, top_k_weights, restore_idx, num_tokens, top_k)

        # Expected: weighted sum of duplicated tokens
        expected = (
            hidden_states.unsqueeze(1).expand(-1, top_k, -1).float()
            * top_k_weights.unsqueeze(-1).float()
        ).sum(dim=1).half()

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-3)

    @pytest.mark.parametrize("num_tokens", [32, 128])
    def test_roundtrip_with_scaling(self, num_tokens: int):
        """Permute -> scale by 2 -> unpermute should equal 2 * weighted input."""
        num_experts, top_k, hidden_dim = 8, 2, 512
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        top_k_indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")
        top_k_weights = torch.softmax(
            torch.randn(num_tokens, top_k, device="cuda", dtype=torch.float32), dim=-1,
        ).half()

        permuted, _, sorted_idx, restore_idx = permute_tokens(
            hidden_states, top_k_indices, num_experts,
        )

        # Scale expert outputs by 2
        expert_outputs = permuted * 2.0

        output = unpermute_tokens(expert_outputs, top_k_weights, restore_idx, num_tokens, top_k)

        expected = 2.0 * (
            hidden_states.unsqueeze(1).expand(-1, top_k, -1).float()
            * top_k_weights.unsqueeze(-1).float()
        ).sum(dim=1).half()

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-3)

    def test_roundtrip_no_nan(self):
        """Roundtrip should not produce NaN or Inf."""
        num_tokens, num_experts, top_k, hidden_dim = 256, 64, 4, 1024
        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        top_k_indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")
        top_k_weights = torch.softmax(
            torch.randn(num_tokens, top_k, device="cuda", dtype=torch.float32), dim=-1,
        ).half()

        permuted, _, _, restore_idx = permute_tokens(hidden_states, top_k_indices, num_experts)
        output = unpermute_tokens(permuted, top_k_weights, restore_idx, num_tokens, top_k)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestPermuteModelConfigs:
    """Test permute/unpermute with realistic MoE model configurations."""

    @pytest.mark.parametrize("model_name,num_tokens", [
        ("mixtral-8x7b", 128),
        ("mixtral-8x7b", 1024),
        ("qwen2-moe-57b", 128),
        ("deepseek-v3", 32),
    ])
    def test_model_config_roundtrip(self, model_name: str, num_tokens: int):
        """Roundtrip works for real model dimensions."""
        cfg = MODEL_CONFIGS[model_name]
        hidden_dim = cfg["hidden_dim"]
        num_experts = cfg["num_experts"]
        top_k = cfg["top_k"]

        hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
        top_k_indices = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")
        top_k_weights = torch.softmax(
            torch.randn(num_tokens, top_k, device="cuda", dtype=torch.float32), dim=-1,
        ).half()

        permuted, offsets, _, restore_idx = permute_tokens(
            hidden_states, top_k_indices, num_experts,
        )

        assert permuted.shape == (num_tokens * top_k, hidden_dim)
        assert offsets[-1].item() == num_tokens * top_k

        output = unpermute_tokens(permuted, top_k_weights, restore_idx, num_tokens, top_k)

        assert output.shape == (num_tokens, hidden_dim)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
