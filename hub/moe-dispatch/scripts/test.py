# /// script
# dependencies = ["kernels", "torch"]
# ///
"""Post-publish GPU test: load moe_dispatch from the Hub and run a forward pass.

Run by the test-kernel workflow on an HF Jobs GPU flavor. It pulls the exact
artifacts the build workflow uploaded, so it validates the published kernel end
to end (not the local source tree).
"""
import torch
from kernels import get_kernel

# trust_remote_code=True is required until bassrehab is a trusted publisher.
moe = get_kernel("bassrehab/moe-dispatch", version=0, trust_remote_code=True)

nt, h, f, ne, tk = 64, 256, 512, 8, 2
dev, dt = "cuda", torch.float16
hs = torch.randn(nt, h, dtype=dt, device=dev)
rw = torch.randn(ne, h, dtype=dt, device=dev)
wg = torch.randn(ne, f, h, dtype=dt, device=dev) * 0.02
wu = torch.randn(ne, f, h, dtype=dt, device=dev) * 0.02
wd = torch.randn(ne, h, f, dtype=dt, device=dev) * 0.02

out, idx, weights = moe.fused_moe_forward(hs, rw, wg, wu, wd, ne, tk, gating="softmax")

assert out.shape == (nt, h), out.shape
assert idx.shape == (nt, tk), idx.shape
assert torch.isfinite(out).all(), "non-finite output"
print("OK", tuple(out.shape))
