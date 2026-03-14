"""
Triton kernel for MoE router: gating score computation and top-k expert selection.

The router takes token hidden states, projects them to expert logits via a learned
weight matrix, applies a gating function (softmax or sigmoid), and selects the
top-k experts per token along with their gating weights.

This is the first stage of the MoE dispatch pipeline. The kernel fuses:
1. Router projection (hidden_states @ router_weight.T)
2. Gating function (stable softmax or sigmoid)
3. Top-K selection with iterative max-mask

Performance characteristics:
- Router projection is compute-bound for large batch sizes
- Top-K selection is memory-bound (small compute per token)
- Softmax requires two passes (max + exp-sum) for numerical stability

The router projection is handled by PyTorch matmul (leveraging cuBLAS tensor cores),
while the gating + top-k selection is fused into a single Triton kernel.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def _softmax_topk_kernel(
    Logits,          # Input logits pointer: (num_tokens, num_experts)
    TopK_Values,     # Output top-k values pointer: (num_tokens, TOP_K)
    TopK_Indices,    # Output top-k indices pointer: (num_tokens, TOP_K)
    num_tokens,      # Number of tokens
    stride_logits_t, # Stride between rows in logits
    stride_val_t,    # Stride between rows in values output
    stride_idx_t,    # Stride between rows in indices output
    NUM_EXPERTS: tl.constexpr,  # Number of experts (constexpr for compile-time shapes)
    TOP_K: tl.constexpr,        # Number of experts to select per token
):
    """
    Fused softmax + top-k selection kernel.

    Each program instance handles one token: computes stable softmax over
    expert logits, then iteratively selects the top-k experts by finding
    the max, storing it, and masking it out.

    Numerical stability: uses online stable softmax (subtract max before exp).
    """
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    # Load all expert logits for this token
    offsets = tl.arange(0, NUM_EXPERTS)
    logits = tl.load(Logits + pid * stride_logits_t + offsets, mask=offsets < NUM_EXPERTS, other=float('-inf'))
    logits = logits.to(tl.float32)

    # Stable softmax: subtract max, then exp, then normalize
    max_val = tl.max(logits, axis=0)
    logits_shifted = logits - max_val
    exp_logits = tl.exp(logits_shifted)
    sum_exp = tl.sum(exp_logits, axis=0)
    scores = exp_logits / sum_exp

    # Iterative top-k: find max, store, mask, repeat
    for k in tl.static_range(TOP_K):
        # Find max score and its index
        cur_max = tl.max(scores, axis=0)
        # Create mask for the max value (first occurrence)
        is_max = scores == cur_max
        # Get the index of the max: use argmax
        max_idx = tl.argmax(scores, axis=0)

        # Store value and index
        tl.store(TopK_Values + pid * stride_val_t + k, cur_max)
        tl.store(TopK_Indices + pid * stride_idx_t + k, max_idx)

        # Mask out selected expert for next iteration
        mask = offsets == max_idx
        scores = tl.where(mask, 0.0, scores)


@triton.jit
def _sigmoid_topk_kernel(
    Logits,          # Input logits pointer: (num_tokens, num_experts)
    TopK_Values,     # Output top-k values pointer: (num_tokens, TOP_K)
    TopK_Indices,    # Output top-k indices pointer: (num_tokens, TOP_K)
    num_tokens,      # Number of tokens
    stride_logits_t, # Stride between rows in logits
    stride_val_t,    # Stride between rows in values output
    stride_idx_t,    # Stride between rows in indices output
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
):
    """
    Fused sigmoid gating + top-k selection kernel.

    Similar to softmax variant but uses sigmoid instead and normalizes
    the selected weights to sum to 1.
    """
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    # Load all expert logits for this token
    offsets = tl.arange(0, NUM_EXPERTS)
    logits = tl.load(Logits + pid * stride_logits_t + offsets, mask=offsets < NUM_EXPERTS, other=float('-inf'))
    logits = logits.to(tl.float32)

    # Sigmoid gating
    scores = 1.0 / (1.0 + tl.exp(-logits))

    # Iterative top-k selection (before normalization)
    # We store raw sigmoid scores, then normalize in a second pass
    weight_sum = 0.0
    for k in tl.static_range(TOP_K):
        cur_max = tl.max(scores, axis=0)
        max_idx = tl.argmax(scores, axis=0)

        tl.store(TopK_Values + pid * stride_val_t + k, cur_max)
        tl.store(TopK_Indices + pid * stride_idx_t + k, max_idx)

        weight_sum += cur_max

        # Mask out selected expert
        mask = offsets == max_idx
        scores = tl.where(mask, 0.0, scores)

    # Normalize weights to sum to 1
    inv_sum = 1.0 / (weight_sum + 1e-6)
    for k in tl.static_range(TOP_K):
        val = tl.load(TopK_Values + pid * stride_val_t + k)
        tl.store(TopK_Values + pid * stride_val_t + k, val * inv_sum)


def moe_router(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int,
    gating: str = "softmax",
    router_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute MoE routing using Triton kernel for gating + top-k selection.

    The router projection (matmul) is done via PyTorch (cuBLAS) since it's
    already highly optimized. The gating + top-k is fused in a Triton kernel.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Input token representations. Shape: (num_tokens, hidden_dim).
    router_weight : torch.Tensor
        Router projection matrix. Shape: (num_experts, hidden_dim).
    top_k : int
        Number of experts to select per token.
    gating : str
        Gating function: "softmax" or "sigmoid".
    router_bias : torch.Tensor, optional
        Router bias. Shape: (num_experts,).

    Returns
    -------
    top_k_indices : torch.Tensor
        Selected expert indices. Shape: (num_tokens, top_k). dtype: int64.
    top_k_weights : torch.Tensor
        Gating weights for selected experts. Shape: (num_tokens, top_k).
    router_logits : torch.Tensor
        Raw router logits. Shape: (num_tokens, num_experts).
    """
    assert hidden_states.is_cuda, "Input must be on CUDA device"
    assert router_weight.is_cuda, "Router weight must be on CUDA device"
    assert hidden_states.shape[-1] == router_weight.shape[-1], (
        f"Hidden dim mismatch: {hidden_states.shape[-1]} vs {router_weight.shape[-1]}"
    )

    num_tokens = hidden_states.shape[0]
    num_experts = router_weight.shape[0]

    assert top_k <= num_experts, f"top_k ({top_k}) must be <= num_experts ({num_experts})"

    # Router projection via cuBLAS: (num_tokens, hidden_dim) @ (hidden_dim, num_experts)
    router_logits = torch.nn.functional.linear(
        hidden_states.float(), router_weight.float(), router_bias
    ).to(hidden_states.dtype)

    # Allocate outputs
    top_k_values = torch.empty(num_tokens, top_k, device=hidden_states.device, dtype=torch.float32)
    top_k_indices = torch.empty(num_tokens, top_k, device=hidden_states.device, dtype=torch.int64)

    # NUM_EXPERTS must be a power of 2 for Triton constexpr range
    # Pad to next power of 2 if needed (logits are masked with -inf for padding)
    num_experts_padded = triton.next_power_of_2(num_experts)

    # Pad logits if num_experts is not a power of 2
    if num_experts_padded != num_experts:
        padding = torch.full(
            (num_tokens, num_experts_padded - num_experts),
            float('-inf'),
            device=hidden_states.device,
            dtype=router_logits.dtype,
        )
        logits_padded = torch.cat([router_logits, padding], dim=-1)
    else:
        logits_padded = router_logits

    # Launch Triton kernel: one program per token
    grid = (num_tokens,)

    if gating == "softmax":
        _softmax_topk_kernel[grid](
            logits_padded,
            top_k_values,
            top_k_indices,
            num_tokens,
            logits_padded.stride(0),
            top_k_values.stride(0),
            top_k_indices.stride(0),
            NUM_EXPERTS=num_experts_padded,
            TOP_K=top_k,
        )
    elif gating == "sigmoid":
        _sigmoid_topk_kernel[grid](
            logits_padded,
            top_k_values,
            top_k_indices,
            num_tokens,
            logits_padded.stride(0),
            top_k_values.stride(0),
            top_k_indices.stride(0),
            NUM_EXPERTS=num_experts_padded,
            TOP_K=top_k,
        )
    else:
        raise ValueError(f"Unknown gating function: {gating}. Use 'softmax' or 'sigmoid'.")

    return top_k_indices, top_k_values.to(hidden_states.dtype), router_logits


def moe_router_torch(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int,
    gating: str = "softmax",
    router_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of MoE routing.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Input token representations. Shape: (num_tokens, hidden_dim).
    router_weight : torch.Tensor
        Router projection matrix. Shape: (num_experts, hidden_dim).
    top_k : int
        Number of experts to select per token.
    gating : str
        Gating function: "softmax" or "sigmoid".
    router_bias : torch.Tensor, optional
        Router bias. Shape: (num_experts,).

    Returns
    -------
    top_k_indices : torch.Tensor
        Selected expert indices. Shape: (num_tokens, top_k).
    top_k_weights : torch.Tensor
        Gating weights. Shape: (num_tokens, top_k).
    router_logits : torch.Tensor
        Raw logits. Shape: (num_tokens, num_experts).
    """
    router_logits = torch.nn.functional.linear(
        hidden_states.float(), router_weight.float(), router_bias
    )

    if gating == "softmax":
        scores = torch.softmax(router_logits, dim=-1)
    elif gating == "sigmoid":
        scores = torch.sigmoid(router_logits)
    else:
        raise ValueError(f"Unknown gating: {gating}")

    top_k_weights, top_k_indices = torch.topk(scores, k=top_k, dim=-1)

    if gating == "sigmoid":
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)

    router_logits = router_logits.to(hidden_states.dtype)
    top_k_weights = top_k_weights.to(hidden_states.dtype)

    return top_k_indices, top_k_weights, router_logits
