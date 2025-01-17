"""
Fused SwiGLU activation kernel implemented in Triton.

SwiGLU (Swish-Gated Linear Unit) is used in LLaMA, PaLM, and other modern LLMs.
It's a variant of GLU that uses SiLU (Swish) as the activation function.

Full SwiGLU formula: y = silu(x @ W_gate) * (x @ W_up)

This kernel implements the fused activation part (after the linear projections):
y = silu(gate) * up

Where silu(x) = x * sigmoid(x)

Performance characteristics:
- Memory-bound operation (low arithmetic intensity ~1.3 FLOPs/byte)
- Fusing silu and multiply saves one memory round-trip
- Uses numerically stable sigmoid implementation

Reference: https://arxiv.org/abs/2204.02311 (PaLM paper)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    Gate,        # Gate tensor pointer (after W_gate projection)
    Up,          # Up tensor pointer (after W_up projection)
    Out,         # Output tensor pointer
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for fused SwiGLU activation.

    Computes: out = silu(gate) * up = gate * sigmoid(gate) * up

    Memory access pattern:
    - Each program processes BLOCK_SIZE elements
    - Coalesced reads of gate and up tensors
    - Coalesced write of output
    - All operations are elementwise

    Arithmetic intensity:
    - Reads: N (gate) + N (up) = 2N elements
    - Writes: N (out) = N elements
    - FLOPs per element:
        - sigmoid: ~4 ops (exp, add, div) - but optimized
        - gate * sigmoid(gate): 1 mul
        - * up: 1 mul
    - Total: ~4 FLOPs per element
    - AI ≈ 4 / (3 * 2) = 0.67 for FP16 (memory-bound)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load gate and up values
    gate = tl.load(Gate + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Up + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute SiLU(gate) = gate * sigmoid(gate)
    # sigmoid(x) = 1 / (1 + exp(-x))
    # For numerical stability, use: sigmoid(x) = 0.5 * (1 + tanh(x/2))
    # But the standard form works fine with Triton's exp implementation
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate

    # Compute output: silu(gate) * up
    out = silu_gate * up

    # Store output
    tl.store(Out + offsets, out, mask=mask)


def swiglu_fused(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """
    Fused SwiGLU activation.

    Computes: y = silu(gate) * up

    This is the activation portion of SwiGLU, applied after the linear
    projections. The full SwiGLU in a transformer FFN is:
        y = silu(x @ W_gate) * (x @ W_up)

    This kernel fuses the silu and elementwise multiply, saving one
    memory round-trip compared to: F.silu(gate) * up

    Args:
        gate: Gate tensor of shape (...,), result of x @ W_gate projection.
        up: Up tensor of same shape as gate, result of x @ W_up projection.

    Returns:
        Output tensor of same shape as inputs.

    Example:
        >>> # In a transformer FFN:
        >>> gate = x @ W_gate  # Shape: (batch, seq, ffn_dim)
        >>> up = x @ W_up      # Shape: (batch, seq, ffn_dim)
        >>> y = swiglu_fused(gate, up)
    """
    assert gate.is_cuda, "Gate must be on CUDA device"
    assert up.is_cuda, "Up must be on CUDA device"
    assert gate.shape == up.shape, f"Shape mismatch: gate={gate.shape}, up={up.shape}"

    # Flatten for kernel
    original_shape = gate.shape
    gate_flat = gate.view(-1)
    up_flat = up.view(-1)
    n_elements = gate_flat.numel()

    # Allocate output
    out_flat = torch.empty_like(gate_flat)

    # Launch kernel
    # FIXME: smaller block size might be better for small tensors
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _swiglu_kernel[grid](
        gate_flat,
        up_flat,
        out_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_flat.view(original_shape)


def swiglu_torch(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch reference implementation of SwiGLU activation.

    Used for correctness validation.
    """
    return torch.nn.functional.silu(gate) * up


class SwiGLU(torch.nn.Module):
    """
    SwiGLU activation module using Triton kernel.

    This module implements just the activation part. For a full SwiGLU FFN layer,
    you would combine this with linear projections:

        class SwiGLUFFN(nn.Module):
            def __init__(self, hidden_dim, ffn_dim):
                super().__init__()
                self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
                self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
                self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)
                self.act = SwiGLU()

            def forward(self, x):
                return self.w_down(self.act(self.w_gate(x), self.w_up(x)))

    Example:
        >>> act = SwiGLU()
        >>> gate = torch.randn(2, 1024, 11008, device='cuda', dtype=torch.float16)
        >>> up = torch.randn_like(gate)
        >>> y = act(gate, up)
    """

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return swiglu_fused(gate, up)


# For convenience, also expose a fused linear + swiglu variant
@triton.jit
def _swiglu_fused_kernel_with_bias(
    Gate,        # Gate tensor pointer
    Up,          # Up tensor pointer
    BiasGate,    # Gate bias pointer (can be None via mask)
    BiasUp,      # Up bias pointer (can be None via mask)
    Out,         # Output tensor pointer
    n_elements,  # Total elements
    hidden_dim,  # For bias indexing
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SwiGLU kernel variant that can optionally add biases.

    Computes: out = silu(gate + bias_gate) * (up + bias_up)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load gate and up values
    gate = tl.load(Gate + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Up + offsets, mask=mask, other=0.0).to(tl.float32)

    # Add biases if present
    if HAS_BIAS:
        bias_idx = offsets % hidden_dim
        bias_gate = tl.load(BiasGate + bias_idx, mask=mask, other=0.0).to(tl.float32)
        bias_up = tl.load(BiasUp + bias_idx, mask=mask, other=0.0).to(tl.float32)
        gate = gate + bias_gate
        up = up + bias_up

    # Compute SiLU(gate) * up
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    out = silu_gate * up

    tl.store(Out + offsets, out, mask=mask)


def swiglu_with_bias(
    gate: torch.Tensor,
    up: torch.Tensor,
    bias_gate: torch.Tensor,
    bias_up: torch.Tensor,
) -> torch.Tensor:
    """
    Fused SwiGLU with bias addition.

    Computes: y = silu(gate + bias_gate) * (up + bias_up)

    Useful when linear layers have biases.
    """
    assert gate.shape == up.shape
    assert gate.shape[-1] == bias_gate.shape[0] == bias_up.shape[0]

    original_shape = gate.shape
    gate_flat = gate.view(-1)
    up_flat = up.view(-1)
    n_elements = gate_flat.numel()
    hidden_dim = gate.shape[-1]

    out_flat = torch.empty_like(gate_flat)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _swiglu_fused_kernel_with_bias[grid](
        gate_flat,
        up_flat,
        bias_gate,
        bias_up,
        out_flat,
        n_elements,
        hidden_dim,
        HAS_BIAS=True,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_flat.view(original_shape)
