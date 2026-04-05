"""
Triton kernels for MoE token permutation and unpermutation.

These kernels reorder tokens between token-major layout (batch_size, hidden_dim)
and expert-contiguous layout where all tokens for expert 0 come first, then
expert 1, etc. This enables coalesced memory access during expert GEMMs.

The pipeline:
1. **permute_tokens**: Given top-k expert assignments per token, compute a
   histogram of tokens per expert, prefix-sum for offsets, and scatter tokens
   into expert-contiguous order.
2. **unpermute_tokens**: After expert FFNs, scatter outputs back to original
   token order and apply weighted combination across top-k experts.

Performance characteristics:
- Both operations are pure data movement (memory-bound)
- Arithmetic intensity ≈ 0 FLOPs/byte for permute, ~1 FLOP/byte for unpermute (weight mul + add)
- Target: ≥80% of peak memory bandwidth on A100 (2 TB/s)
- Key optimization: coalesced reads/writes via BLOCK_D tiling over hidden dimension
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _permute_kernel(
    Input,              # (num_tokens, hidden_dim)
    Output,             # (num_tokens * top_k, hidden_dim)
    TopK_Indices,       # (num_tokens, top_k) - expert assignments
    SortedIdx,          # (num_tokens * top_k,) - sort order by expert
    num_tokens,
    hidden_dim,
    top_k,
    stride_in_t,       # input stride between tokens
    stride_out_t,      # output stride between tokens
    stride_idx_t,      # top_k_indices stride between tokens
    BLOCK_D: tl.constexpr,   # tile size over hidden dimension
):
    """
    Permute tokens to expert-contiguous layout.

    Each program handles one (token_slot, hidden_dim_block) tile, where
    token_slot indexes into the sorted output and hidden_dim_block tiles
    over the hidden dimension for coalesced access.

    The kernel reads from the original token (determined by SortedIdx)
    and writes to the contiguous output position.
    """
    # pid_t: which slot in the sorted output
    # pid_d: which block of hidden_dim
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_t >= num_tokens * top_k:
        return

    # Which position in the flattened (num_tokens * top_k) array does this
    # sorted slot correspond to?
    src_flat_idx = tl.load(SortedIdx + pid_t)

    # subhadipmitra, 2026-03-14: the flat index encodes both token_id and which
    # of its top_k experts this copy belongs to. We only need token_id for the
    # permute (to know which row of Input to read from). The k_idx matters for
    # the unpermute when we need to look up the corresponding gating weight.
    src_token = src_flat_idx // top_k

    # Compute hidden dim offsets for this block
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < hidden_dim

    # Read from original token position
    in_ptrs = Input + src_token * stride_in_t + d_offs
    vals = tl.load(in_ptrs, mask=d_mask, other=0.0)

    # Write to sorted output position — this write is contiguous across programs
    # which is the whole point of the permutation
    out_ptrs = Output + pid_t * stride_out_t + d_offs
    tl.store(out_ptrs, vals, mask=d_mask)


@triton.jit
def _unpermute_kernel(
    Input,              # (num_tokens * top_k, hidden_dim) - expert outputs in sorted order
    Output,             # (num_tokens, hidden_dim) - combined output
    SortedIdx,          # (num_tokens * top_k,) - maps sorted position -> flat (token, k) index
    TopK_Weights,       # (num_tokens, top_k) - gating weights
    num_tokens,
    hidden_dim,
    top_k,
    stride_in_t,
    stride_out_t,
    stride_w_t,         # weights stride between tokens
    BLOCK_D: tl.constexpr,
):
    """
    Unpermute expert outputs back to original token order with weighted combination.

    Each program handles one (token, hidden_dim_block) tile. For each token,
    it gathers the top_k expert outputs from their sorted positions, multiplies
    by gating weights, and accumulates.
    """
    pid_t = tl.program_id(0)   # token index
    pid_d = tl.program_id(1)   # hidden dim block

    if pid_t >= num_tokens:
        return

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < hidden_dim

    # subhadipmitra, 2026-03-14: accumulate in FP32 even though inputs are FP16.
    # The weighted sum across top_k experts can lose significant precision in FP16,
    # especially when one expert dominates (weight ~0.99) and others are small (~0.01).
    # Saw ~2% max error without this on Mixtral configs
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for k in range(top_k):
        # Load gating weight for this (token, k)
        weight = tl.load(TopK_Weights + pid_t * stride_w_t + k).to(tl.float32)

        # subhadipmitra, 2026-03-14: restore_idx maps flat_idx -> sorted_position,
        # so we can look up where each (token, k) pair ended up after the sort.
        # This avoids needing an inverse sort at unpermute time
        flat_idx = pid_t * top_k + k
        sorted_pos = tl.load(SortedIdx + flat_idx)

        # Load expert output from sorted position
        in_ptrs = Input + sorted_pos * stride_in_t + d_offs
        expert_out = tl.load(in_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        acc += weight * expert_out

    # Write combined output
    out_ptrs = Output + pid_t * stride_out_t + d_offs
    tl.store(out_ptrs, acc.to(tl.float16), mask=d_mask)


def compute_permutation_indices(
    top_k_indices: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the permutation mapping from top-k expert assignments.

    Given per-token expert assignments, computes:
    - sorted_idx: maps sorted position -> original flat (token * top_k + k) index
    - expert_offsets: cumulative boundaries per expert in the sorted array
    - restore_idx: maps original flat index -> sorted position (inverse of sorted_idx)

    Parameters
    ----------
    top_k_indices : torch.Tensor
        Expert assignments. Shape: (num_tokens, top_k).
    num_experts : int
        Total number of experts.

    Returns
    -------
    sorted_idx : torch.Tensor
        Sort order. Shape: (num_tokens * top_k,).
    expert_offsets : torch.Tensor
        Expert boundaries. Shape: (num_experts + 1,).
    restore_idx : torch.Tensor
        Inverse permutation. Shape: (num_tokens * top_k,).
    """
    num_tokens, top_k = top_k_indices.shape
    device = top_k_indices.device

    # Flatten expert assignments
    flat_experts = top_k_indices.reshape(-1)

    # subhadipmitra, 2026-03-14: stable=True preserves token order within each
    # expert bucket — same rationale as in the reference implementation
    _, sorted_idx = flat_experts.sort(stable=True)

    # Compute expert offsets via histogram
    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    expert_counts.scatter_add_(0, flat_experts.long(), torch.ones_like(flat_experts, dtype=torch.int64))
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = expert_counts.cumsum(dim=0)

    # Compute inverse mapping: restore_idx[flat_idx] = sorted_position
    # subhadipmitra, 2026-03-14: the inverse permutation is needed by the unpermute
    # kernel — given a (token, k) pair, we need to know where it lives in the
    # sorted array to read the expert output from the right place
    restore_idx = torch.empty_like(sorted_idx)
    restore_idx[sorted_idx] = torch.arange(len(sorted_idx), device=device)

    return sorted_idx, expert_offsets, restore_idx


def permute_tokens(
    hidden_states: torch.Tensor,
    top_k_indices: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Permute tokens to expert-contiguous layout using Triton kernel.

    Each token is duplicated top_k times (once per selected expert) and reordered
    so all tokens for the same expert are contiguous in memory.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Input tokens. Shape: (num_tokens, hidden_dim).
    top_k_indices : torch.Tensor
        Expert assignments per token. Shape: (num_tokens, top_k).
    num_experts : int
        Total number of experts.

    Returns
    -------
    permuted : torch.Tensor
        Tokens in expert-contiguous order. Shape: (num_tokens * top_k, hidden_dim).
    expert_offsets : torch.Tensor
        Start index per expert. Shape: (num_experts + 1,).
    sorted_idx : torch.Tensor
        Permutation mapping. Shape: (num_tokens * top_k,).
    restore_idx : torch.Tensor
        Inverse permutation. Shape: (num_tokens * top_k,).
    """
    assert hidden_states.is_cuda, "Input must be on CUDA"
    num_tokens, hidden_dim = hidden_states.shape
    top_k = top_k_indices.shape[1]

    # Compute permutation indices
    sorted_idx, expert_offsets, restore_idx = compute_permutation_indices(
        top_k_indices, num_experts,
    )

    # Allocate output
    permuted = torch.empty(num_tokens * top_k, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)

    # subhadipmitra, 2026-03-14: cap BLOCK_D at 1024 to keep register pressure
    # reasonable. For hidden_dim=4096 this means 4 blocks per token, each doing
    # coalesced 1024-element reads/writes. Benchmarked 512 vs 1024 vs 2048 on A100,
    # 1024 was the sweet spot for bandwidth utilization
    BLOCK_D = min(triton.next_power_of_2(hidden_dim), 1024)
    grid = (num_tokens * top_k, triton.cdiv(hidden_dim, BLOCK_D))

    _permute_kernel[grid](
        hidden_states,
        permuted,
        top_k_indices,
        sorted_idx,
        num_tokens,
        hidden_dim,
        top_k,
        hidden_states.stride(0),
        permuted.stride(0),
        top_k_indices.stride(0),
        BLOCK_D=BLOCK_D,
    )

    return permuted, expert_offsets, sorted_idx, restore_idx


def unpermute_tokens(
    expert_outputs: torch.Tensor,
    top_k_weights: torch.Tensor,
    restore_idx: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """
    Scatter expert outputs back to original token order with weighted combination.

    Parameters
    ----------
    expert_outputs : torch.Tensor
        Expert FFN outputs in sorted order. Shape: (num_tokens * top_k, hidden_dim).
    top_k_weights : torch.Tensor
        Gating weights. Shape: (num_tokens, top_k).
    restore_idx : torch.Tensor
        Maps original flat (token*top_k+k) index -> sorted position.
        Shape: (num_tokens * top_k,).
    num_tokens : int
        Number of original tokens.
    top_k : int
        Number of experts per token.

    Returns
    -------
    output : torch.Tensor
        Combined output. Shape: (num_tokens, hidden_dim).
    """
    assert expert_outputs.is_cuda, "Input must be on CUDA"
    hidden_dim = expert_outputs.shape[-1]

    output = torch.empty(num_tokens, hidden_dim, device=expert_outputs.device, dtype=expert_outputs.dtype)

    BLOCK_D = min(triton.next_power_of_2(hidden_dim), 1024)
    grid = (num_tokens, triton.cdiv(hidden_dim, BLOCK_D))

    _unpermute_kernel[grid](
        expert_outputs,
        output,
        restore_idx,
        top_k_weights,
        num_tokens,
        hidden_dim,
        top_k,
        expert_outputs.stride(0),
        output.stride(0),
        top_k_weights.stride(0),
        BLOCK_D=BLOCK_D,
    )

    return output


def permute_tokens_torch(
    hidden_states: torch.Tensor,
    top_k_indices: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of token permutation.
    """
    num_tokens, hidden_dim = hidden_states.shape
    top_k = top_k_indices.shape[1]

    flat_experts = top_k_indices.reshape(-1)
    _, sorted_idx = flat_experts.sort(stable=True)

    expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=hidden_states.device)
    expert_counts.scatter_add_(0, flat_experts.long(), torch.ones_like(flat_experts, dtype=torch.int64))
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=hidden_states.device)
    expert_offsets[1:] = expert_counts.cumsum(dim=0)

    # Expand tokens for top_k duplication
    expanded = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)
    permuted = expanded[sorted_idx]

    restore_idx = torch.empty_like(sorted_idx)
    restore_idx[sorted_idx] = torch.arange(len(sorted_idx), device=sorted_idx.device)

    return permuted, expert_offsets, sorted_idx, restore_idx


def unpermute_tokens_torch(
    expert_outputs: torch.Tensor,
    top_k_weights: torch.Tensor,
    restore_idx: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """
    PyTorch reference implementation of token unpermutation.
    """
    hidden_dim = expert_outputs.shape[-1]

    # Gather by restore_idx: for each (token, k), find its sorted position
    gathered = expert_outputs[restore_idx]
    gathered = gathered.view(num_tokens, top_k, hidden_dim)

    weights = top_k_weights.unsqueeze(-1).float()
    combined = (gathered.float() * weights).sum(dim=1).to(expert_outputs.dtype)

    return combined
