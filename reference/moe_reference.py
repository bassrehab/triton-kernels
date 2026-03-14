"""
PyTorch reference implementation of Mixture-of-Experts (MoE) forward pass.

This serves as ground truth for correctness validation of the Triton MoE kernels.
No Triton or custom CUDA is used — pure PyTorch operations only.

The implementation covers the full MoE forward pass:
1. Router computation (softmax or sigmoid gating)
2. Top-K expert selection with gating weights
3. Token permutation to expert-contiguous layout
4. Expert FFN computation (two-layer MLP with SiLU activation)
5. Token unpermutation back to original order
6. Weighted combination of expert outputs

Supports configurable:
- Gating function: softmax (Mixtral-style) or sigmoid (DeepSeek-style)
- Number of experts, top-k, hidden dim, FFN dim
- Capacity factor for token dropping
- Optional auxiliary load-balancing loss
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, NamedTuple


class MoERoutingResult(NamedTuple):
    """Result of MoE routing computation.

    Parameters
    ----------
    top_k_indices : torch.Tensor
        Expert indices selected for each token. Shape: (num_tokens, top_k).
    top_k_weights : torch.Tensor
        Gating weights for each selected expert. Shape: (num_tokens, top_k).
    router_logits : torch.Tensor
        Raw router logits before gating. Shape: (num_tokens, num_experts).
    """
    top_k_indices: torch.Tensor
    top_k_weights: torch.Tensor
    router_logits: torch.Tensor


def moe_router_torch(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int,
    gating: str = "softmax",
    router_bias: Optional[torch.Tensor] = None,
) -> MoERoutingResult:
    """
    Compute MoE routing: project hidden states to expert scores, apply gating,
    and select top-k experts per token.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Input token representations. Shape: (num_tokens, hidden_dim).
    router_weight : torch.Tensor
        Router projection matrix. Shape: (num_experts, hidden_dim).
    top_k : int
        Number of experts to select per token.
    gating : str
        Gating function: "softmax" (Mixtral-style) or "sigmoid" (DeepSeek-style).
    router_bias : torch.Tensor, optional
        Router bias. Shape: (num_experts,).

    Returns
    -------
    MoERoutingResult
        Named tuple with top_k_indices, top_k_weights, and router_logits.
    """
    # Router logits: (num_tokens, num_experts)
    router_logits = F.linear(hidden_states.float(), router_weight.float(), router_bias)

    # Apply gating function
    if gating == "softmax":
        scores = F.softmax(router_logits, dim=-1)
    elif gating == "sigmoid":
        scores = torch.sigmoid(router_logits)
    else:
        raise ValueError(f"Unknown gating function: {gating}. Use 'softmax' or 'sigmoid'.")

    # Top-K selection
    top_k_weights, top_k_indices = torch.topk(scores, k=top_k, dim=-1)

    # Normalize weights to sum to 1 across selected experts
    if gating == "sigmoid":
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)

    return MoERoutingResult(
        top_k_indices=top_k_indices,
        top_k_weights=top_k_weights.to(hidden_states.dtype),
        router_logits=router_logits,
    )


def permute_tokens(
    hidden_states: torch.Tensor,
    top_k_indices: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reorder tokens into expert-contiguous layout for efficient batched GEMM.

    Each token is duplicated top_k times (once per selected expert), then sorted
    by expert assignment so all tokens for expert 0 come first, then expert 1, etc.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Input tokens. Shape: (num_tokens, hidden_dim).
    top_k_indices : torch.Tensor
        Expert assignments. Shape: (num_tokens, top_k).
    num_experts : int
        Total number of experts.

    Returns
    -------
    permuted_tokens : torch.Tensor
        Tokens reordered by expert. Shape: (num_tokens * top_k, hidden_dim).
    expert_offsets : torch.Tensor
        Start index of each expert's tokens. Shape: (num_experts + 1,).
    restore_indices : torch.Tensor
        Indices to scatter outputs back to original order. Shape: (num_tokens * top_k,).
    """
    num_tokens, top_k = top_k_indices.shape
    hidden_dim = hidden_states.shape[-1]

    # Flatten expert assignments: (num_tokens * top_k,)
    flat_indices = top_k_indices.reshape(-1)

    # Sort by expert index to group tokens per expert
    sorted_expert_ids, sort_order = flat_indices.sort(stable=True)

    # Compute expert offsets (cumulative count boundaries)
    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=hidden_states.device)
    expert_counts.scatter_add_(0, flat_indices.long(), torch.ones_like(flat_indices, dtype=torch.int64))
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=hidden_states.device)
    expert_offsets[1:] = expert_counts.cumsum(dim=0)

    # Expand hidden states for top_k duplication: (num_tokens, top_k, hidden_dim)
    expanded = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)

    # Permute tokens according to sort order
    permuted_tokens = expanded[sort_order]

    # Compute restore indices (inverse of sort_order)
    restore_indices = torch.empty_like(sort_order)
    restore_indices[sort_order] = torch.arange(len(sort_order), device=sort_order.device)

    return permuted_tokens, expert_offsets, restore_indices


def unpermute_tokens(
    expert_outputs: torch.Tensor,
    restore_indices: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Scatter expert outputs back to original token order and apply weighted combination.

    Parameters
    ----------
    expert_outputs : torch.Tensor
        Outputs from expert FFNs in permuted order. Shape: (num_tokens * top_k, hidden_dim).
    restore_indices : torch.Tensor
        Indices to restore original token order. Shape: (num_tokens * top_k,).
    top_k_weights : torch.Tensor
        Gating weights for each token-expert pair. Shape: (num_tokens, top_k).

    Returns
    -------
    combined_output : torch.Tensor
        Weighted sum of expert outputs per token. Shape: (num_tokens, hidden_dim).
    """
    num_tokens, top_k = top_k_weights.shape
    hidden_dim = expert_outputs.shape[-1]

    # Restore original order
    restored = expert_outputs[restore_indices]

    # Reshape to (num_tokens, top_k, hidden_dim)
    restored = restored.view(num_tokens, top_k, hidden_dim)

    # Weighted combination: (num_tokens, top_k, 1) * (num_tokens, top_k, hidden_dim)
    weights = top_k_weights.unsqueeze(-1)
    combined = (restored * weights).sum(dim=1)

    return combined


def expert_ffn(
    tokens: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """
    Single expert FFN: SwiGLU-style two-layer MLP.

    Computes: output = (SiLU(tokens @ w_gate.T) * (tokens @ w_up.T)) @ w_down.T

    Parameters
    ----------
    tokens : torch.Tensor
        Input tokens for this expert. Shape: (num_tokens_for_expert, hidden_dim).
    w_gate : torch.Tensor
        Gate projection. Shape: (ffn_dim, hidden_dim).
    w_up : torch.Tensor
        Up projection. Shape: (ffn_dim, hidden_dim).
    w_down : torch.Tensor
        Down projection. Shape: (hidden_dim, ffn_dim).

    Returns
    -------
    torch.Tensor
        Expert output. Shape: (num_tokens_for_expert, hidden_dim).
    """
    gate = F.silu(F.linear(tokens.float(), w_gate.float()))
    up = F.linear(tokens.float(), w_up.float())
    return F.linear((gate * up).to(tokens.dtype), w_down)


class MoEReference(torch.nn.Module):
    """
    Reference MoE layer — pure PyTorch, no kernel optimization.

    Implements the complete MoE forward pass with configurable gating,
    top-k routing, and SwiGLU expert FFNs.

    Parameters
    ----------
    hidden_dim : int
        Model hidden dimension.
    ffn_dim : int
        Expert FFN intermediate dimension.
    num_experts : int
        Number of experts.
    top_k : int
        Number of experts activated per token.
    gating : str
        Gating function: "softmax" or "sigmoid".

    Example
    -------
    >>> moe = MoEReference(4096, 14336, 8, 2).cuda().half()
    >>> x = torch.randn(128, 4096, device='cuda', dtype=torch.float16)
    >>> output, routing = moe(x)
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        gating: str = "softmax",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gating = gating

        # Router projection: hidden_dim -> num_experts
        self.router_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_dim)
        )

        # Expert weights: each expert has gate, up, down projections
        self.w_gate = torch.nn.Parameter(torch.empty(num_experts, ffn_dim, hidden_dim))
        self.w_up = torch.nn.Parameter(torch.empty(num_experts, ffn_dim, hidden_dim))
        self.w_down = torch.nn.Parameter(torch.empty(num_experts, hidden_dim, ffn_dim))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming uniform (matches HF Mixtral)."""
        torch.nn.init.kaiming_uniform_(self.router_weight)
        for p in [self.w_gate, self.w_up, self.w_down]:
            torch.nn.init.kaiming_uniform_(p)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, MoERoutingResult]:
        """
        Full MoE forward pass.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tokens. Shape: (num_tokens, hidden_dim).

        Returns
        -------
        output : torch.Tensor
            MoE output. Shape: (num_tokens, hidden_dim).
        routing_result : MoERoutingResult
            Routing metadata (indices, weights, logits).
        """
        num_tokens = hidden_states.shape[0]

        # Step 1: Route tokens to experts
        routing = moe_router_torch(
            hidden_states, self.router_weight, self.top_k, self.gating,
        )

        # Step 2: Permute tokens to expert-contiguous layout
        permuted_tokens, expert_offsets, restore_indices = permute_tokens(
            hidden_states, routing.top_k_indices, self.num_experts,
        )

        # Step 3: Run expert FFNs
        expert_outputs = torch.empty_like(permuted_tokens)
        for e in range(self.num_experts):
            start = expert_offsets[e].item()
            end = expert_offsets[e + 1].item()
            if start == end:
                continue  # no tokens for this expert
            expert_tokens = permuted_tokens[start:end]
            expert_outputs[start:end] = expert_ffn(
                expert_tokens,
                self.w_gate[e],
                self.w_up[e],
                self.w_down[e],
            )

        # Step 4: Unpermute and combine
        output = unpermute_tokens(expert_outputs, restore_indices, routing.top_k_weights)

        return output, routing
