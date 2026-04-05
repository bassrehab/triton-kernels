"""
Fused MoE dispatch kernels — the core of this project.

This module provides a fused gate+up GEMM kernel and a top-level entry point
that eliminates most intermediate global memory traffic in the MoE forward pass.

Fusion strategy:
    Fused gate+up GEMM with in-register SiLU activation:
        - Reads permuted tokens ONCE from global memory
        - Computes both gate and up projections in the same tile loop
        - Applies SiLU(gate) * up in registers before writing intermediate
        - Saves one full read+write of (total_tokens, ffn_dim) vs unfused

    Down projection + unpermute remain separate kernels:
        - Down projection uses the grouped GEMM kernel from expert_gemm.py
        - Unpermute uses the existing scatter kernel from permute.py

    # subhadipmitra, 2026-03-15: I tried fusing the down projection with a
    # weighted scatter (atomic_add to output positions), but Triton doesn't
    # support scalar indexing into 2D accumulators (acc[m, :] fails to compile).
    # A persistent-kernel approach could work but adds significant complexity.
    # The gate+up fusion alone saves ~35% of global memory traffic, which is
    # the majority of the win — the down+scatter fusion would add another ~15%.
    # Leaving that as future work once Triton adds better indexing support.

Memory traffic comparison for Mixtral-8x7B (4096 tokens, hidden=4096, ffn=14336):
    Unfused:  ~1.2 GB  (permuted + gate_out + up_out + intermediate + expert_out + final)
    Fused:    ~0.8 GB  (permuted + intermediate + expert_out + final)
    Savings:  ~35% global memory traffic reduction

The router projection and permutation remain unfused — the router matmul is already
optimal via cuBLAS, and the permutation requires a global sort that can't be fused
with the GEMM without a persistent kernel approach.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple

from triton_kernels.moe.router import moe_router
from triton_kernels.moe.permute import permute_tokens, unpermute_tokens
from triton_kernels.moe.expert_gemm import grouped_gemm, _build_block_schedule


# subhadipmitra, 2026-03-15: same BLOCK_M=64 constraint as the unfused grouped GEMM —
# must match the block schedule. Autotune only BLOCK_N and BLOCK_K
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['N', 'K'],
)
@triton.jit
def _fused_gate_up_kernel(
    # Inputs
    A,                  # permuted tokens: (total_tokens, K)
    B_gate,             # gate weights flattened: (num_experts * N, K)
    B_up,               # up weights flattened: (num_experts * N, K)
    # Output
    C,                  # intermediate output: (total_tokens, N) — SiLU(gate) * up
    # Block scheduling
    ExpertOffsets,
    BlockToExpert,
    BlockToM,
    # Dimensions
    N,                  # ffn_dim
    K,                  # hidden_dim
    num_blocks,
    # Strides
    stride_a_t, stride_a_k,
    stride_bg_n, stride_bg_k,
    stride_bu_n, stride_bu_k,
    stride_c_t, stride_c_n,
    # Tile sizes
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr = 64,
):
    """
    Fused gate + up projection with in-register SiLU activation.

    For each tile, computes:
        gate_tile = A_tile @ B_gate_tile.T
        up_tile   = A_tile @ B_up_tile.T
        output    = SiLU(gate_tile) * up_tile

    The key optimization is that both GEMMs share the same A_tile loads —
    we read A once and compute two matmuls. The SiLU and multiply happen
    in registers before writing to global memory, eliminating two intermediate
    tensors (gate_out and up_out) from global memory entirely.
    """
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid >= num_blocks:
        return

    expert_id = tl.load(BlockToExpert + pid)
    m_start = tl.load(BlockToM + pid)
    expert_token_start = tl.load(ExpertOffsets + expert_id)
    global_m_start = expert_token_start + m_start

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A (shared between gate and up)
    a_ptrs = A + (global_m_start + offs_m[:, None]) * stride_a_t + offs_k[None, :] * stride_a_k

    # subhadipmitra, 2026-03-15: the key insight for the pointer arithmetic here
    # is that expert weights are flattened to (num_experts * N, K). So expert e's
    # weight row j starts at offset (e * N + j). Gate and up weights use the same
    # layout but are separate tensors
    bg_ptrs = B_gate + (expert_id * N + offs_n[None, :]) * stride_bg_n + offs_k[:, None] * stride_bg_k
    bu_ptrs = B_up + (expert_id * N + offs_n[None, :]) * stride_bu_n + offs_k[:, None] * stride_bu_k

    # Bounds check for partial tiles
    expert_token_end = tl.load(ExpertOffsets + expert_id + 1)
    expert_num_tokens = expert_token_end - expert_token_start
    valid_m = m_start + offs_m < expert_num_tokens

    # subhadipmitra, 2026-03-15: two accumulators in registers — this is the whole
    # point of the fusion. Both GEMMs read the same A tile from L2 cache, so the
    # second one is essentially free from a memory bandwidth perspective. On A100
    # the L2 hit rate for the shared A tiles should be near 100% since BLOCK_M*BLOCK_K
    # = 64*32 = 2KB fits comfortably in the 40MB L2
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # Load A tile (shared between both projections)
        a_mask = valid_m[:, None] & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load gate and up weight tiles
        b_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        b_gate = tl.load(bg_ptrs, mask=b_mask, other=0.0)
        b_up = tl.load(bu_ptrs, mask=b_mask, other=0.0)

        # Accumulate both projections from the same A tile
        acc_gate += tl.dot(a, b_gate, out_dtype=tl.float32)
        acc_up += tl.dot(a, b_up, out_dtype=tl.float32)

        a_ptrs += BLOCK_K * stride_a_k
        bg_ptrs += BLOCK_K * stride_bg_k
        bu_ptrs += BLOCK_K * stride_bu_k

    # subhadipmitra, 2026-03-15: SiLU + multiply in FP32 registers. This is where
    # we save the most memory traffic — instead of writing gate_out (total_tokens * ffn_dim * 2B)
    # and up_out (same) to global memory then reading them back, we just compute
    # the result in-place. For Mixtral (14336 ffn_dim, 4096 tokens * 2 top_k),
    # this saves ~470 MB of global memory traffic per forward pass
    silu_gate = acc_gate * tl.sigmoid(acc_gate)  # SiLU(x) = x * sigmoid(x)
    result = silu_gate * acc_up

    # Write fused output
    c_ptrs = C + (global_m_start + offs_m[:, None]) * stride_c_t + offs_n[None, :] * stride_c_n
    c_mask = valid_m[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, result.to(tl.float16), mask=c_mask)


def fused_expert_ffn(
    permuted_tokens: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    expert_offsets: torch.Tensor,
    num_experts: int,
    top_k_weights: torch.Tensor,
    sorted_idx: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """
    Expert FFN with fused gate+up kernel, regular down projection + unpermute.

    The gate+up fusion eliminates two intermediate buffers (gate_out, up_out)
    from global memory by computing SiLU(gate) * up in registers.

    Parameters
    ----------
    permuted_tokens : torch.Tensor
        Expert-contiguous tokens. Shape: (total_tokens, hidden_dim).
    w_gate : torch.Tensor
        Gate weights. Shape: (num_experts, ffn_dim, hidden_dim).
    w_up : torch.Tensor
        Up weights. Shape: (num_experts, ffn_dim, hidden_dim).
    w_down : torch.Tensor
        Down weights. Shape: (num_experts, hidden_dim, ffn_dim).
    expert_offsets : torch.Tensor
        Expert boundaries. Shape: (num_experts + 1,).
    num_experts : int
        Number of experts.
    top_k_weights : torch.Tensor
        Gating weights. Shape: (num_tokens, top_k).
    sorted_idx : torch.Tensor
        Maps sorted position -> original flat (token*top_k+k). Shape: (total_tokens,).
    num_tokens : int
        Original number of tokens (before top_k expansion).
    top_k : int
        Number of experts per token.

    Returns
    -------
    output : torch.Tensor
        Final MoE output. Shape: (num_tokens, hidden_dim).
    """
    total_tokens, hidden_dim = permuted_tokens.shape
    ffn_dim = w_gate.shape[1]

    # Flatten weights for contiguous kernel access
    w_gate_flat = w_gate.reshape(num_experts * ffn_dim, hidden_dim).contiguous()
    w_up_flat = w_up.reshape(num_experts * ffn_dim, hidden_dim).contiguous()

    # Build block schedule (shared between fused gate+up and down GEMM)
    block_to_expert, block_to_m, num_blocks = _build_block_schedule(
        expert_offsets, num_experts, BLOCK_M=64,
    )

    if num_blocks == 0:
        return torch.zeros(num_tokens, hidden_dim, device=permuted_tokens.device, dtype=permuted_tokens.dtype)

    # --- Fused gate + up + SiLU kernel ---
    # subhadipmitra, 2026-03-15: this single kernel replaces what was previously
    # two grouped_gemm() calls + a SiLU + multiply. The intermediate tensor is
    # the only thing written to global memory here
    intermediate = torch.empty(total_tokens, ffn_dim, device=permuted_tokens.device, dtype=permuted_tokens.dtype)

    def grid(META):
        return (num_blocks, triton.cdiv(ffn_dim, META['BLOCK_N']))

    _fused_gate_up_kernel[grid](
        permuted_tokens,
        w_gate_flat, w_up_flat,
        intermediate,
        expert_offsets, block_to_expert, block_to_m,
        ffn_dim, hidden_dim, num_blocks,
        permuted_tokens.stride(0), permuted_tokens.stride(1),
        w_gate_flat.stride(0), w_gate_flat.stride(1),
        w_up_flat.stride(0), w_up_flat.stride(1),
        intermediate.stride(0), intermediate.stride(1),
    )

    # --- Down projection via regular grouped GEMM ---
    expert_output = grouped_gemm(intermediate, w_down, expert_offsets, num_experts)

    # --- Unpermute + weighted combine via regular kernel ---
    # subhadipmitra, 2026-03-15: compute restore_idx from sorted_idx for unpermute.
    # sorted_idx maps sorted_pos -> flat_idx, restore_idx is the inverse
    restore_idx = torch.empty_like(sorted_idx)
    restore_idx[sorted_idx] = torch.arange(len(sorted_idx), device=sorted_idx.device)

    output = unpermute_tokens(expert_output, top_k_weights, restore_idx, num_tokens, top_k)

    return output


def fused_moe_forward(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    num_experts: int,
    top_k: int,
    gating: str = "softmax",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Complete fused MoE forward pass.

    This is the top-level entry point that runs the full pipeline:
        1. Router projection + gating + top-k selection (Triton kernel)
        2. Token permutation to expert-contiguous layout (Triton kernel)
        3. Fused gate+up GEMM with in-register SiLU (Triton kernel)
        4. Down projection via grouped GEMM (Triton kernel)
        5. Unpermute + weighted combine (Triton kernel)

    Total: 5 kernel launches vs 7 in the unfused pipeline (router, permute,
    gate GEMM, up GEMM, SiLU+mul, down GEMM, unpermute).

    Parameters
    ----------
    hidden_states : torch.Tensor
        Input tokens. Shape: (num_tokens, hidden_dim).
    router_weight : torch.Tensor
        Router projection. Shape: (num_experts, hidden_dim).
    w_gate : torch.Tensor
        Expert gate weights. Shape: (num_experts, ffn_dim, hidden_dim).
    w_up : torch.Tensor
        Expert up weights. Shape: (num_experts, ffn_dim, hidden_dim).
    w_down : torch.Tensor
        Expert down weights. Shape: (num_experts, hidden_dim, ffn_dim).
    num_experts : int
        Number of experts.
    top_k : int
        Experts per token.
    gating : str
        "softmax" or "sigmoid".

    Returns
    -------
    output : torch.Tensor
        MoE output. Shape: (num_tokens, hidden_dim).
    top_k_indices : torch.Tensor
        Expert assignments. Shape: (num_tokens, top_k).
    top_k_weights : torch.Tensor
        Gating weights. Shape: (num_tokens, top_k).
    """
    num_tokens = hidden_states.shape[0]

    # Step 1: Route
    top_k_indices, top_k_weights, _ = moe_router(
        hidden_states, router_weight, top_k, gating,
    )

    # Step 2: Permute
    permuted, expert_offsets, sorted_idx, restore_idx = permute_tokens(
        hidden_states, top_k_indices, num_experts,
    )

    # Steps 3-5: Fused gate+up, then down, then unpermute
    output = fused_expert_ffn(
        permuted, w_gate, w_up, w_down,
        expert_offsets, num_experts,
        top_k_weights, sorted_idx,
        num_tokens, top_k,
    )

    return output, top_k_indices, top_k_weights
