# Triton Kernels

High-performance GPU kernels for LLM inference, implemented in [OpenAI Triton](https://triton-lang.org/).

This repository provides educational, well-documented implementations of common transformer operations optimized for inference. Each kernel includes roofline analysis explaining *why* the optimization works at the hardware level.

## Why Custom Kernels?

LLM inference is **memory-bandwidth bound**. A 7B parameter model in FP16 requires loading 14GB of weights for every forward pass. On an A100 (2TB/s bandwidth), this takes ~7ms—while the actual computation is <0.1ms.

Custom kernels help by:

- **Fusing operations** to reduce memory round-trips
- **Quantizing weights** to reduce memory traffic
- **Maximizing bandwidth utilization** through memory-aware access patterns

## Kernels

| Kernel | Description | Speedup vs PyTorch |
|--------|-------------|-------------------|
| [`rmsnorm`](triton_kernels/rmsnorm.py) | RMSNorm with FP32 accumulation | **8.1x** |
| [`rmsnorm_residual_fused`](triton_kernels/rmsnorm.py) | Fused RMSNorm + residual add | **6.0x** |
| [`swiglu_fused`](triton_kernels/swiglu.py) | Fused SiLU-gated linear unit | **1.6x** |
| [`int8_gemm`](triton_kernels/quantized_matmul.py) | W8A16 quantized matrix multiply | ~1.0x (2x memory savings) |
| [`fused_moe_forward`](triton_kernels/moe/fused_moe.py) | Fused MoE dispatch (router + experts) | **up to 9.1x** |

## Installation

```bash
# Clone the repository
git clone https://github.com/bassrehab/triton-kernels.git
cd triton-kernels

# Install in development mode
pip install -e .

# Or with all dependencies (testing + benchmarking)
pip install -e ".[all]"
```

**Requirements:**
- Python 3.10+
- PyTorch 2.1+
- Triton 3.0+
- NVIDIA GPU (compute capability 8.0+) or AMD GPU (MI250X/MI300X via ROCm)

## Quick Start

```python
import torch
from triton_kernels import (
    rmsnorm,
    rmsnorm_residual_fused,
    swiglu_fused,
    int8_gemm,
    quantize_weight_per_channel,
)

# ==============================================================================
# Fused RMSNorm + Residual
# ==============================================================================
# Common pattern in transformer blocks: normalize(x + residual)
x = torch.randn(1, 2048, 4096, device='cuda', dtype=torch.float16)
residual = torch.randn_like(x)
weight = torch.ones(4096, device='cuda', dtype=torch.float16)

# Fused: avoids materializing x + residual intermediate tensor
y = rmsnorm_residual_fused(x, residual, weight, eps=1e-6)

# ==============================================================================
# Fused SwiGLU
# ==============================================================================
# Used in LLaMA, Mistral, and other modern LLMs
gate = torch.randn(1, 2048, 11008, device='cuda', dtype=torch.float16)
up = torch.randn_like(gate)

# Fused: silu(gate) * up in one kernel
y = swiglu_fused(gate, up)

# ==============================================================================
# INT8 Quantized GEMM
# ==============================================================================
# W8A16: INT8 weights, FP16 activations (2x memory reduction for weights)
x = torch.randn(1, 2048, 4096, device='cuda', dtype=torch.float16)
weight_fp16 = torch.randn(11008, 4096, dtype=torch.float16)

# Quantize weights (typically done once at load time)
weight_int8, scale = quantize_weight_per_channel(weight_fp16)
weight_int8 = weight_int8.cuda()
scale = scale.cuda()

# INT8 GEMM with on-the-fly dequantization
y = int8_gemm(x, weight_int8, scale)

# ==============================================================================
# Fused MoE Dispatch (Mixtral / DeepSeek-V3 / Qwen2-MoE style)
# ==============================================================================
# Complete MoE forward pass: router → permute → fused expert FFN → unpermute
from triton_kernels import fused_moe_forward

num_experts, top_k = 8, 2
hidden_dim, ffn_dim = 4096, 14336
x = torch.randn(128, hidden_dim, device='cuda', dtype=torch.float16)
router_weight = torch.randn(num_experts, hidden_dim, device='cuda', dtype=torch.float16)
w_gate = torch.randn(num_experts, ffn_dim, hidden_dim, device='cuda', dtype=torch.float16) * 0.02
w_up = torch.randn(num_experts, ffn_dim, hidden_dim, device='cuda', dtype=torch.float16) * 0.02
w_down = torch.randn(num_experts, hidden_dim, ffn_dim, device='cuda', dtype=torch.float16) * 0.02

output, expert_indices, expert_weights = fused_moe_forward(
    x, router_weight, w_gate, w_up, w_down,
    num_experts, top_k, gating="softmax",
)
```

## Drop-in Modules

For easy integration with existing models:

```python
from triton_kernels import TritonRMSNorm, SwiGLU, Int8Linear

# Replace torch.nn.RMSNorm
norm = TritonRMSNorm(hidden_size=4096, eps=1e-6).cuda()

# Replace F.silu(gate) * up
activation = SwiGLU()

# Replace nn.Linear with INT8 quantized version
linear = Int8Linear.from_linear(pretrained_linear_layer)
```

## Benchmarks

Run benchmarks on your hardware:

```bash
# Individual kernels
python -m benchmarks.bench_rmsnorm
python -m benchmarks.bench_swiglu
python -m benchmarks.bench_quantized_matmul

# MoE dispatch benchmarks
python -m benchmarks.bench_moe_dispatch --model mixtral-8x7b --batch-sizes 32,128,512,2048
python -m benchmarks.bench_moe_dispatch --model deepseek-v3 --batch-sizes 32,128,512 --skip-reference

# MoE roofline analysis
python -m benchmarks.roofline.moe_roofline --model mixtral-8x7b --num-tokens 512

# Full roofline analysis (generates plots and analysis doc)
python -m benchmarks.full_roofline --output-dir docs/figures
```

### Benchmark Results (A100-SXM4-40GB)

Tested with LLaMA 7B-style dimensions (hidden_dim=4096, ffn_dim=11008, seq_len=2048).

| Kernel | Latency (ms) | Bandwidth (GB/s) | % of Peak | Speedup |
|--------|-------------|------------------|-----------|---------|
| RMSNorm (PyTorch) | 0.30 | 168 | 11% | 1.0x |
| RMSNorm (Triton) | 0.04 | 1365 | 88% | **8.1x** |
| RMSNorm+Residual (PyTorch) | 0.32 | 266 | 17% | 1.0x |
| RMSNorm+Residual (Triton Fused) | 0.05 | 1285 | 83% | **6.0x** |
| SwiGLU (PyTorch) | 0.18 | 1251 | 80% | 1.0x |
| SwiGLU (Triton Fused) | 0.11 | 1223 | 79% | **1.6x** |
| FP16 GEMM (cuBLAS) | 0.76 | 200 | - | 1.0x |
| INT8 GEMM (Triton) | 0.09 | 480 | 31% | ~1.0x |
| INT8 GEMM (cuBLAS, M=2048) | 0.73 | 146 | - | **1.04x** |

*Peak bandwidth: 1555 GB/s. INT8 GEMM provides 2x memory savings for weights.*

### MoE Dispatch Results (A100-SXM4-80GB)

Fused MoE dispatch kernel benchmarked against PyTorch reference (loop-over-experts) and [Megablocks](https://github.com/stanford-futuredata/megablocks) (CUDA-optimized baseline).

**Mixtral-8x7B** (8 experts, top-2, hidden=4096, ffn=14336):

| Tokens | PyTorch Ref | Megablocks | Triton Fused | Speedup vs PyTorch | vs Megablocks |
|--------|------------|------------|--------------|-------------------|---------------|
| 32     | 10.44 ms   | 2.78 ms    | **2.13 ms**  | **4.9x**          | **131%**      |
| 128    | 13.14 ms   | 2.77 ms    | **2.27 ms**  | **5.8x**          | **124%**      |
| 512    | 25.92 ms   | 3.57 ms    | 3.99 ms      | **6.5x**          | 89%           |
| 2048   | 66.22 ms   | 9.08 ms    | 16.48 ms     | **4.0x**          | 56%           |

Our Triton kernel **beats the CUDA-optimized Megablocks** at inference-relevant batch sizes (≤128 tokens) and achieves 89% at 512 tokens — using **zero CUDA code**. Cross-platform validated on AMD MI300X (162/162 tests pass).

![MoE Roofline — Mixtral-8x7B](docs/figures/moe_roofline_mixtral.png)

See [docs/moe_dispatch.md](docs/moe_dispatch.md) for the full technical writeup with roofline analysis and design decisions.

## Roofline Analysis

![Roofline Plot](docs/figures/roofline_all.png)

The roofline model shows where each kernel sits relative to hardware limits:

- **Below the diagonal**: Memory-bound (benefit from fusion/quantization)
- **On the plateau**: Compute-bound (benefit from faster arithmetic)

Most LLM inference operations are memory-bound, which is why our optimizations focus on reducing memory traffic rather than raw FLOPS.

See [docs/ROOFLINE_ANALYSIS.md](docs/ROOFLINE_ANALYSIS.md) for detailed analysis and [docs/INT8_GEMM_INVESTIGATION.md](docs/INT8_GEMM_INVESTIGATION.md) for the INT8 performance investigation.

## Key Insights

### 1. Fusion Wins Big for Memory-Bound Operations

RMSNorm reads and writes the entire tensor. PyTorch launches multiple small kernels with intermediate tensors. Triton fuses everything into one kernel, achieving **88% of peak bandwidth**—an **8x speedup**.

### 2. Quantization is About Memory, Not Compute

Loading INT8 weights instead of FP16 halves memory traffic. However, INT8 tensor cores require quantizing FP16 activations on-the-fly, which adds overhead. **The main value of W8A16 quantization is 2x memory savings**, enabling larger models to fit in GPU memory.

### 3. Bandwidth Utilization Matters More Than FLOPS

Most "optimizations" in LLM inference are really about using the memory bus efficiently. Our Triton kernels achieve 80-88% of peak bandwidth—near optimal. PyTorch baselines often achieve only 10-20% due to kernel launch overhead and intermediate tensors.

## Project Structure

```
triton-kernels/
├── triton_kernels/              # Main package
│   ├── rmsnorm.py               # RMSNorm + fused variants
│   ├── swiglu.py                # SwiGLU activation
│   ├── quantization.py          # INT8 quantization utilities
│   ├── quantized_matmul.py      # W8A16 GEMM kernel
│   └── moe/                     # MoE dispatch kernels
│       ├── router.py            # Softmax/sigmoid gating + top-k
│       ├── permute.py           # Token permute/unpermute
│       ├── expert_gemm.py       # Block-scheduled grouped GEMM
│       └── fused_moe.py         # Fused gate+up kernel + entry point
├── reference/
│   └── moe_reference.py         # PyTorch MoE ground truth
├── benchmarks/                  # Benchmark suite
│   ├── bench_rmsnorm.py
│   ├── bench_swiglu.py
│   ├── bench_quantized_matmul.py
│   ├── bench_moe_dispatch.py    # MoE benchmarks (vs Megablocks)
│   ├── roofline/
│   │   └── moe_roofline.py      # Per-stage MoE roofline analysis
│   ├── full_roofline.py         # Combined analysis
│   └── utils.py                 # GPU detection, roofline plotting
├── tests/                       # Test suite
│   ├── test_rmsnorm.py
│   ├── test_swiglu.py
│   ├── test_quantization.py
│   ├── test_quantized_matmul.py
│   └── test_moe_dispatch.py     # 162 MoE tests
├── docs/
│   ├── ROOFLINE_ANALYSIS.md     # Performance analysis
│   ├── moe_dispatch.md          # MoE technical writeup
│   └── figures/                 # Generated plots
├── pyproject.toml               # Package configuration
└── README.md
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_rmsnorm.py -v

# Run with coverage
pytest tests/ --cov=triton_kernels
```

## Limitations

- **Not production-ready**: These are educational implementations. For production, consider [FlashAttention](https://github.com/Dao-AILab/flash-attention), [vLLM](https://github.com/vllm-project/vllm), or [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
- **Cross-platform**: Tested on NVIDIA A100 and AMD MI300X via Triton's ROCm backend. Performance optimized for NVIDIA; AMD correctness validated.
- **No attention kernel**: A simplified fused attention is a stretch goal; FlashAttention is significantly more complex.

## References

- [Making Deep Learning Go Brrrr (Horace He)](https://horace.io/brrr_intro.html) - Essential reading on GPU optimization
- [FlashAttention (Dao et al.)](https://arxiv.org/abs/2205.14135) - IO-aware attention algorithm
- [Triton Documentation](https://triton-lang.org/) - Official Triton docs
- [RMSNorm (Zhang & Sennrich)](https://arxiv.org/abs/1910.07467) - RMSNorm paper
- [PaLM (Chowdhery et al.)](https://arxiv.org/abs/2204.02311) - SwiGLU activation
- [LLM.int8() (Dettmers et al.)](https://arxiv.org/abs/2208.07339) - INT8 quantization for LLMs
- [MegaBlocks (Gale et al.)](https://arxiv.org/abs/2211.15841) - Efficient sparse MoE training
- [Mixtral of Experts (Jiang et al.)](https://arxiv.org/abs/2401.04088) - Sparse MoE architecture
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - 256-expert MoE with sigmoid gating

## Author

**Subhadip Mitra** - [contact@subhadipmitra.com](mailto:contact@subhadipmitra.com)

## License

MIT
