---
license: mit
tags:
- kernel
- triton
- moe
- mixture-of-experts
- llm-inference
---

# Fused MoE Dispatch (Triton, universal)

A cross-platform, Triton-native **Mixture-of-Experts dispatch** kernel: the full
sparse-expert forward pass in one fused pipeline, with no vendor-specific code.

```
tokens → router scores → top-k gating → permute → fused gate/up GEMM (SiLU)
       → grouped down GEMM → unpermute + weighted combine → output
```

MoE dispatch is the piece of the inference stack that is still mostly CUDA-only
and vendor-locked. This kernel is pure Triton, so a single **universal** build
runs on NVIDIA (SM80/SM90) and AMD (MI300X) - validated for correctness on both.

## Usage

```python
import torch
from kernels import get_kernel

# trust_remote_code=True is required until the publisher is on the Hub
# trusted-publisher list.
moe = get_kernel("bassrehab/moe-dispatch", version=1, trust_remote_code=True)

num_tokens, hidden, ffn, num_experts, top_k = 4096, 4096, 14336, 8, 2
dev, dtype = "cuda", torch.float16

hidden_states = torch.randn(num_tokens, hidden, dtype=dtype, device=dev)
router_weight = torch.randn(num_experts, hidden, dtype=dtype, device=dev)
w_gate = torch.randn(num_experts, ffn, hidden, dtype=dtype, device=dev)
w_up   = torch.randn(num_experts, ffn, hidden, dtype=dtype, device=dev)
w_down = torch.randn(num_experts, hidden, ffn, dtype=dtype, device=dev)

out, top_k_indices, top_k_weights = moe.fused_moe_forward(
    hidden_states, router_weight, w_gate, w_up, w_down,
    num_experts=num_experts, top_k=top_k, gating="softmax",  # or "sigmoid"
)
```

## Public API

The versioned surface (everything exported in `__all__`):

| Symbol | Purpose |
|--------|---------|
| `fused_moe_forward` | End-to-end MoE forward (router → combine). The headline entry point. |
| `fused_expert_ffn` | Fused gate/up + down + unpermute for a pre-permuted batch. |
| `moe_router` | Router projection + gating + top-k selection. |
| `permute_tokens` / `unpermute_tokens` | Token ↔ expert-contiguous layout conversion. |
| `grouped_gemm` | Variable-size grouped GEMM across experts. |
| `expert_ffn_triton` | Standalone expert FFN (unfused reference path). |

## Supported hardware

| Backend | Status |
|---------|--------|
| NVIDIA A100 / H100 (SM80/SM90) | Primary target |
| AMD MI300X (ROCm via Triton) | Correctness-validated |

## Tensor shapes

| Tensor | Shape |
|--------|-------|
| `hidden_states` | `(num_tokens, hidden_dim)` |
| `router_weight` | `(num_experts, hidden_dim)` |
| `w_gate`, `w_up` | `(num_experts, ffn_dim, hidden_dim)` |
| `w_down` | `(num_experts, hidden_dim, ffn_dim)` |

Benchmarked against the reference configurations of Mixtral-8x7B / 8x22B,
DeepSeek-V3, and Qwen2-MoE-57B. All tensors default to FP16 (BF16 where the
hardware supports it).

## Status

**Version 1.** Load with `version=1`. The public API is stable within the major
version; pin an explicit `revision` if you need bit-for-bit reproducibility.

## References

- **Paper:** Mitra, S. *Cross-Platform Fused MoE Dispatch in Triton: Portable
  Expert Routing Without CUDA.* [arXiv:2605.23911](https://arxiv.org/abs/2605.23911)
- **Deep-dive:** [Fused MoE Dispatch in Triton](https://subhadipmitra.com/blog/2026/fused-moe-dispatch-triton/)
  - design decisions and roofline analysis
- **Source & benchmarks:** <https://github.com/bassrehab/triton-kernels>
