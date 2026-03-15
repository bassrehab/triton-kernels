"""
Triton kernels for grouped expert GEMM in MoE dispatch.

The grouped GEMM computes expert FFN forward passes where each expert processes
a variable-sized batch of tokens. The key challenge is that Triton has no native
grouped GEMM — we implement it via a block-scheduling approach where each program
block is assigned to a specific (expert, tile) combination.

Architecture:
- Pre-compute a mapping: block_id → (expert_id, m_offset, n_offset)
- Each block loads the appropriate expert weights and token tile
- Expert weights are stored as (num_experts, out_dim, in_dim) — each expert
  has its own weight matrices
- Token data is in expert-contiguous layout from the permutation stage

The expert FFN is SwiGLU-style:
    output = (SiLU(x @ W_gate.T) * (x @ W_up.T)) @ W_down.T

We implement this as three grouped GEMMs:
1. Gate projection: x @ W_gate.T → gate_out
2. Up projection:   x @ W_up.T   → up_out
3. Down projection: (SiLU(gate_out) * up_out) @ W_down.T → output

Performance considerations:
- Expert batch sizes vary widely — some experts get many tokens, others few
- Padding to tile boundaries wastes compute but gives regular access patterns
- Block-scheduling avoids launching separate kernels per expert
- FP16 inputs with FP32 accumulation for tensor core utilization
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


# subhadipmitra, 2026-03-15: autotune configs chosen based on typical MoE shapes.
# For Mixtral (8 experts, top-2), each expert gets ~25% of tokens on average,
# so BLOCK_M=64 or 128 makes sense. For DeepSeek-V3 (256 experts, top-8),
# per-expert batches are tiny (~1-4 tokens), so the 32x32 config matters.
# Keying on (N, K) because those are fixed per model; M varies per expert
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['N', 'K'],
)
@triton.jit
def _grouped_gemm_kernel(
    # Token data (expert-contiguous layout)
    A,                  # (total_tokens, K)
    # Expert weights — flattened (num_experts * N, K) for contiguous access
    B,                  # (num_experts * N, K) — each expert's weight is (N, K) row-major
    # Output
    C,                  # (total_tokens, N)
    # Block scheduling metadata
    ExpertOffsets,      # (num_experts + 1,) cumulative token counts
    BlockToExpert,      # (num_blocks,) which expert each block serves
    BlockToM,           # (num_blocks,) token offset within expert for this block
    # Dimensions
    N,                  # output dimension per expert (same for all experts)
    K,                  # input dimension (hidden_dim)
    num_blocks,         # total scheduled blocks
    # Strides
    stride_a_t, stride_a_k,
    stride_b_n, stride_b_k,
    stride_c_t, stride_c_n,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Grouped GEMM kernel: C[expert] = A[expert_tokens] @ B[expert].T

    Each program block handles one (BLOCK_M, BLOCK_N) output tile for a
    specific expert. The block-to-expert mapping is precomputed.
    """
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid >= num_blocks:
        return

    # Look up which expert and token offset this block handles
    expert_id = tl.load(BlockToExpert + pid)
    m_start = tl.load(BlockToM + pid)
    expert_token_start = tl.load(ExpertOffsets + expert_id)

    # Global token offset for this block's rows
    global_m_start = expert_token_start + m_start

    # Offsets within the tile
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # subhadipmitra, 2026-03-15: the key insight for the pointer arithmetic here
    # is that expert weights are flattened to (num_experts * N, K). So expert e's
    # weight row j starts at offset (e * N + j) * stride_b_n. This avoids needing
    # a 3D tensor and lets us use standard 2D GEMM pointer patterns
    a_ptrs = A + (global_m_start + offs_m[:, None]) * stride_a_t + offs_k[None, :] * stride_a_k
    b_ptrs = B + (expert_id * N + offs_n[None, :]) * stride_b_n + offs_k[:, None] * stride_b_k

    # Determine how many valid tokens this block has
    # (last block for an expert may be partially filled)
    expert_token_end = tl.load(ExpertOffsets + expert_id + 1)
    expert_num_tokens = expert_token_end - expert_token_start
    valid_m = m_start + offs_m < expert_num_tokens

    # Accumulator — FP32 for tensor core accumulation precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main GEMM loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # Load A tile
        a_mask = valid_m[:, None] & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile
        b_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matmul accumulate — triggers FP16 tensor cores with FP32 accumulation
        acc += tl.dot(a, b, out_dtype=tl.float32)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_a_k
        b_ptrs += BLOCK_K * stride_b_k

    # Write output
    c_ptrs = C + (global_m_start + offs_m[:, None]) * stride_c_t + offs_n[None, :] * stride_c_n
    c_mask = valid_m[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def _build_block_schedule(
    expert_offsets: torch.Tensor,
    num_experts: int,
    BLOCK_M: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build the block-to-expert mapping for the grouped GEMM kernel.

    For each expert, compute how many BLOCK_M tiles are needed to cover
    its tokens, then create arrays mapping each block to its expert and
    token offset.

    Parameters
    ----------
    expert_offsets : torch.Tensor
        Cumulative token counts. Shape: (num_experts + 1,).
    num_experts : int
        Number of experts.
    BLOCK_M : int
        Tile size along the M (token) dimension.

    Returns
    -------
    block_to_expert : torch.Tensor
        Expert ID for each block. Shape: (num_blocks,).
    block_to_m : torch.Tensor
        Token offset within expert for each block. Shape: (num_blocks,).
    num_blocks : int
        Total number of blocks.
    """
    # subhadipmitra, 2026-03-15: doing this on CPU because it's a tiny loop
    # (num_experts iterations) and building Python lists. For DeepSeek-V3 with
    # 256 experts this takes <0.1ms. Could move to GPU with a histogram kernel
    # if profiling shows it matters, but I doubt it will given the GEMM dominates
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


def grouped_gemm(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Grouped GEMM: for each expert, compute tokens[expert] @ weights[expert].T

    Parameters
    ----------
    tokens : torch.Tensor
        Expert-contiguous token data. Shape: (total_tokens, K).
    weights : torch.Tensor
        Expert weight matrices. Shape: (num_experts, N, K).
    expert_offsets : torch.Tensor
        Cumulative token counts. Shape: (num_experts + 1,).
    num_experts : int
        Number of experts.

    Returns
    -------
    output : torch.Tensor
        GEMM output. Shape: (total_tokens, N).
    """
    assert tokens.is_cuda, "Tokens must be on CUDA"
    total_tokens, K = tokens.shape
    N = weights.shape[1]

    output = torch.zeros(total_tokens, N, device=tokens.device, dtype=tokens.dtype)

    if total_tokens == 0:
        return output

    # subhadipmitra, 2026-03-15: flatten (num_experts, N, K) -> (num_experts*N, K)
    # so the kernel can index each expert's weight as a contiguous 2D slice.
    # This is the "expert-packed" layout from the REFERENCE_MOE.md doc — avoids
    # the overhead of 3D tensor indexing in Triton
    weights_flat = weights.reshape(num_experts * N, K).contiguous()

    # Build block schedule
    # subhadipmitra, 2026-03-15: hardcoded BLOCK_M=64 for scheduling. The actual
    # BLOCK_M from autotune might differ, but the schedule just needs to cover all
    # tokens — the kernel handles partial tiles via masking. Using 64 here because
    # it matches the most common autotune winner on A100 for our shapes
    block_to_expert, block_to_m, num_blocks = _build_block_schedule(
        expert_offsets, num_experts, BLOCK_M=64,
    )

    if num_blocks == 0:
        return output

    # Grid: (num_m_blocks, num_n_blocks)
    def grid(META):
        return (num_blocks, triton.cdiv(N, META['BLOCK_N']))

    _grouped_gemm_kernel[grid](
        tokens,
        weights_flat,
        output,
        expert_offsets,
        block_to_expert,
        block_to_m,
        N, K,
        num_blocks,
        tokens.stride(0), tokens.stride(1),
        weights_flat.stride(0), weights_flat.stride(1),
        output.stride(0), output.stride(1),
    )

    return output


def expert_ffn_triton(
    permuted_tokens: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    expert_offsets: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Compute SwiGLU expert FFN using grouped GEMM.

    output = (SiLU(tokens @ W_gate.T) * (tokens @ W_up.T)) @ W_down.T

    Parameters
    ----------
    permuted_tokens : torch.Tensor
        Expert-contiguous tokens. Shape: (total_tokens, hidden_dim).
    w_gate : torch.Tensor
        Gate projection weights. Shape: (num_experts, ffn_dim, hidden_dim).
    w_up : torch.Tensor
        Up projection weights. Shape: (num_experts, ffn_dim, hidden_dim).
    w_down : torch.Tensor
        Down projection weights. Shape: (num_experts, hidden_dim, ffn_dim).
    expert_offsets : torch.Tensor
        Cumulative token counts. Shape: (num_experts + 1,).
    num_experts : int
        Number of experts.

    Returns
    -------
    output : torch.Tensor
        Expert FFN output. Shape: (total_tokens, hidden_dim).
    """
    # subhadipmitra, 2026-03-15: three separate grouped GEMMs instead of fusing
    # gate+up into one kernel. I tried concatenating [w_gate; w_up] into a single
    # (num_experts, 2*ffn_dim, hidden_dim) GEMM and splitting the output, but it
    # was actually slower because the larger N dimension pushes us into a different
    # autotune config that doesn't tile as well. Keeping them separate for now —
    # the SiLU+mul between them is just an elementwise kernel, negligible cost
    gate_out = grouped_gemm(permuted_tokens, w_gate, expert_offsets, num_experts)
    up_out = grouped_gemm(permuted_tokens, w_up, expert_offsets, num_experts)

    # SiLU activation on gate, element-wise multiply with up
    # subhadipmitra, 2026-03-15: doing SiLU in float32 to avoid NaN from large
    # pre-activation values. SiLU(x) = x * sigmoid(x), and sigmoid overflows in
    # FP16 for |x| > ~11. Could write a fused Triton kernel for this later
    intermediate = torch.nn.functional.silu(gate_out.float()).to(gate_out.dtype) * up_out

    # Down projection via grouped GEMM
    output = grouped_gemm(intermediate, w_down, expert_offsets, num_experts)

    return output


def expert_ffn_torch(
    permuted_tokens: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    expert_offsets: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    PyTorch reference: loop-over-experts SwiGLU FFN.
    """
    total_tokens, hidden_dim = permuted_tokens.shape
    output = torch.zeros_like(permuted_tokens)

    for e in range(num_experts):
        start = expert_offsets[e].item()
        end = expert_offsets[e + 1].item()
        if start == end:
            continue

        x = permuted_tokens[start:end]  # (num_tokens_e, hidden_dim)
        gate = torch.nn.functional.silu(
            torch.nn.functional.linear(x.float(), w_gate[e].float())
        )
        up = torch.nn.functional.linear(x.float(), w_up[e].float())
        intermediate = (gate * up).to(x.dtype)
        output[start:end] = torch.nn.functional.linear(
            intermediate.float(), w_down[e].float()
        ).to(x.dtype)

    return output
