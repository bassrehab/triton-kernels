# /// script
# dependencies = ["kernels", "torch"]
# ///
"""Post-publish GPU test: load w4a16 from the Hub and run a forward pass.

Run by the test workflow on an HF Jobs GPU. Pulls the exact published artifacts, so it
validates the published kernel end to end.
"""
import torch
from kernels import get_kernel

# trust_remote_code=True is required until bassrehab is a trusted publisher.
w4a16 = get_kernel("bassrehab/w4a16", version=1, trust_remote_code=True)

K, N, G, M = 4096, 512, 128, 8
packed, scales, zeros = w4a16.quantize_weight_int4_grouped(torch.randn(K, N) * 0.1, G)

x = torch.randn(M, K, dtype=torch.float16, device="cuda")
y = w4a16.w4a16_gemm(x, packed.cuda(), scales.cuda().half(), zeros.cuda().half(), G)

assert y.shape == (M, N), y.shape
assert torch.isfinite(y).all(), "non-finite output"
print("OK", tuple(y.shape))
