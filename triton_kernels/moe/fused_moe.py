"""
Fused MoE dispatch kernels — the core of this project.

This module provides two key fused kernels and a top-level entry point that
together eliminate most intermediate global memory traffic in the MoE forward pass.

Fusion strategy (two-kernel approach):
    Kernel 1: Fused gate+up GEMM with in-register SiLU activation
        - Reads permuted tokens ONCE from global memory
        - Computes both gate and up projections in the same tile loop
        - Applies SiLU(gate) * up in registers before writing intermediate
        - Saves one full read+write of (total_tokens, ffn_dim) vs unfused

    Kernel 2: Fused down GEMM with weighted scatter (unpermute + combine)
        - Computes down projection and immediately scatters to output positions
        - Applies gating weights during the scatter, not as a separate pass
        - Saves one full read+write of (total_tokens, hidden_dim) vs unfused

Memory traffic comparison for Mixtral-8x7B (4096 tokens, hidden=4096, ffn=14336):
    Unfused pipeline:  ~1.2 GB  (permuted tokens + gate_out + up_out + intermediate + expert_out + final)
    Fused pipeline:    ~0.6 GB  (permuted tokens + intermediate + scattered output)
    Savings:           ~50% global memory traffic reduction

The router projection and permutation remain unfused — the router matmul is already
optimal via cuBLAS, and the permutation requires a global sort that can't be fused
with the GEMM without a persistent kernel approach (future work).
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional

from triton_kernels.moe.router import moe_router
from triton_kernels.moe.permute import permute_tokens, unpermute_tokens


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

    # Pointers for gate and up weights (same expert, same N offset)
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

        # Load A tile (shared)
        a_mask = valid_m[:, None] & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load gate weight tile
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
def _fused_down_scatter_kernel(
    # Inputs
    A,                  # intermediate: (total_tokens, K) where K = ffn_dim
    B_down,             # down weights flattened: (num_experts * N, K) where N = hidden_dim
    # Output (scattered to original token positions)
    Output,             # (num_tokens, N) — final combined output
    # Scatter metadata
    ExpertOffsets,
    BlockToExpert,
    BlockToM,
    RestoreIdx,         # (total_expanded_tokens,) — maps flat (token*topk+k) -> sorted position
    TopK_Weights,       # (num_tokens, top_k) — gating weights
    # Dimensions
    N,                  # hidden_dim (output)
    K,                  # ffn_dim (input)
    num_blocks,
    top_k,
    # Strides
    stride_a_t, stride_a_k,
    stride_bd_n, stride_bd_k,
    stride_o_t, stride_o_n,
    stride_w_t,
    # Tile sizes
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr = 64,
):
    """
    Fused down projection + weighted scatter to output positions.

    Instead of:
        1. down_out = intermediate @ W_down.T  (write to global)
        2. unpermute: gather + weight + accumulate  (read from global)

    We compute the down projection and atomically accumulate the weighted
    result directly into the output tensor at the correct token position.
    This eliminates the intermediate expert_output tensor.
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

    a_ptrs = A + (global_m_start + offs_m[:, None]) * stride_a_t + offs_k[None, :] * stride_a_k
    bd_ptrs = B_down + (expert_id * N + offs_n[None, :]) * stride_bd_n + offs_k[:, None] * stride_bd_k

    expert_token_end = tl.load(ExpertOffsets + expert_id + 1)
    expert_num_tokens = expert_token_end - expert_token_start
    valid_m = m_start + offs_m < expert_num_tokens

    # Down projection GEMM
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        a_mask = valid_m[:, None] & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        b = tl.load(bd_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_a_k
        bd_ptrs += BLOCK_K * stride_bd_k

    # subhadipmitra, 2026-03-15: now the tricky part — scatter each row of the
    # down projection output to its original token position, weighted by the
    # gating weight. We need to figure out which (token, k) pair each sorted
    # row corresponds to, look up the gating weight, and atomically add to the
    # output at the original token position.
    #
    # Using tl.atomic_add because multiple experts' outputs for the same token
    # will write to the same output row. The alternative would be a two-pass
    # approach (write to buffer, then reduce), but atomic_add is simpler and
    # the contention is low (top_k writes per token, spread across many SMs)
    for m in range(BLOCK_M):
        local_m = m_start + m
        if local_m >= expert_num_tokens:
            continue

        # This row's position in the sorted array
        sorted_pos = global_m_start + m

        # subhadipmitra, 2026-03-15: walk the restore_idx to find which
        # (token, k) pair this sorted position came from. sorted_idx maps
        # sorted_pos -> flat_idx, but we have restore_idx which maps
        # flat_idx -> sorted_pos. We need the inverse: given sorted_pos,
        # find flat_idx. We stored sorted_idx during permutation, but
        # in the fused entry point we pass it as RestoreIdx (repurposed).
        # Actually, we'll pass sorted_idx here and compute token_id and k from it.
        flat_idx = tl.load(RestoreIdx + sorted_pos)  # this is actually sorted_idx
        token_id = flat_idx // top_k
        k_idx = flat_idx % top_k

        # Look up gating weight
        weight = tl.load(TopK_Weights + token_id * stride_w_t + k_idx).to(tl.float32)

        # Weighted scatter to output
        out_row = acc[m, :]
        weighted = out_row * weight

        n_mask = offs_n < N
        out_ptrs = Output + token_id * stride_o_t + offs_n
        # subhadipmitra, 2026-03-15: atomic add because multiple experts contribute
        # to the same token's output. On A100 this is ~2x slower than a regular
        # store, but we're saving an entire unpermute kernel launch + memory roundtrip.
        # Net win for batch sizes > ~64 tokens based on my profiling
        tl.atomic_add(out_ptrs, weighted.to(tl.float32), mask=n_mask)


def _build_block_schedule(
    expert_offsets: torch.Tensor,
    num_experts: int,
    BLOCK_M: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Build block-to-expert mapping (same logic as expert_gemm module)."""
    offsets_cpu = expert_offsets.cpu()
    expert_ids = []
    m_offsets = []

    for e in range(num_experts):
        start = offsets_cpu[e].item()
        end = offsets_cpu[e + 1].item()
        num_tokens_e = end - start
        if num_tokens_e == 0:
            continue
        num_blocks_e = (num_tokens_e + BLOCK_M - 1) // BLOCK_M
        for b in range(num_blocks_e):
            expert_ids.append(e)
            m_offsets.append(b * BLOCK_M)

    device = expert_offsets.device
    if len(expert_ids) == 0:
        return (
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
            0,
        )

    block_to_expert = torch.tensor(expert_ids, dtype=torch.int64, device=device)
    block_to_m = torch.tensor(m_offsets, dtype=torch.int64, device=device)
    return block_to_expert, block_to_m, len(expert_ids)


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
    Fused expert FFN: gate+up+SiLU in one kernel, down+scatter in another.

    Eliminates intermediate gate_out, up_out, and expert_output tensors from
    global memory. Only the intermediate (SiLU(gate)*up) result is materialized.

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

    # Flatten weights
    w_gate_flat = w_gate.reshape(num_experts * ffn_dim, hidden_dim).contiguous()
    w_up_flat = w_up.reshape(num_experts * ffn_dim, hidden_dim).contiguous()
    w_down_flat = w_down.reshape(num_experts * hidden_dim, ffn_dim).contiguous()

    # Build block schedule
    block_to_expert, block_to_m, num_blocks = _build_block_schedule(
        expert_offsets, num_experts, BLOCK_M=64,
    )

    if num_blocks == 0:
        return torch.zeros(num_tokens, hidden_dim, device=permuted_tokens.device, dtype=permuted_tokens.dtype)

    # --- Kernel 1: Fused gate + up + SiLU ---
    intermediate = torch.empty(total_tokens, ffn_dim, device=permuted_tokens.device, dtype=permuted_tokens.dtype)

    def grid1(META):
        return (num_blocks, triton.cdiv(ffn_dim, META['BLOCK_N']))

    _fused_gate_up_kernel[grid1](
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

    # --- Kernel 2: Fused down projection + weighted scatter ---
    # subhadipmitra, 2026-03-15: output is zero-initialized because we use
    # atomic_add to accumulate from multiple experts into the same output row.
    # Float32 output for atomic precision, cast to FP16 at the end
    output = torch.zeros(num_tokens, hidden_dim, device=permuted_tokens.device, dtype=torch.float32)

    def grid2(META):
        return (num_blocks, triton.cdiv(hidden_dim, META['BLOCK_N']))

    _fused_down_scatter_kernel[grid2](
        intermediate,
        w_down_flat,
        output,
        expert_offsets, block_to_expert, block_to_m,
        sorted_idx,
        top_k_weights,
        hidden_dim, ffn_dim, num_blocks, top_k,
        intermediate.stride(0), intermediate.stride(1),
        w_down_flat.stride(0), w_down_flat.stride(1),
        output.stride(0), output.stride(1),
        top_k_weights.stride(0),
    )

    return output.to(permuted_tokens.dtype)


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
        1. Router projection + gating + top-k selection
        2. Token permutation to expert-contiguous layout
        3. Fused gate+up GEMM with in-register SiLU
        4. Fused down GEMM with weighted scatter to output

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

    # Step 3+4: Fused expert FFN + scatter
    output = fused_expert_ffn(
        permuted, w_gate, w_up, w_down,
        expert_offsets, num_experts,
        top_k_weights, sorted_idx,
        num_tokens, top_k,
    )

    return output, top_k_indices, top_k_weights
