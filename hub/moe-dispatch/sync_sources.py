#!/usr/bin/env python3
"""Regenerate the Hub kernel sources from the canonical triton_kernels/moe package.

The Hugging Face Kernel Hub requires a self-contained package that uses only
*relative* imports (see the kernel-requirements docs: a kernel version is loaded
under a uniquely-mangled module name, so an absolute ``from triton_kernels.moe.X``
would resolve against the *installed* triton_kernels - wrong or absent on a user's
machine - instead of the sibling file shipped in the build).

Rather than maintain a hand-edited second copy that silently drifts from the
canonical kernel, this script copies the source files verbatim and rewrites only
the intra-package imports.

Run from anywhere::

    python hub/moe-dispatch/sync_sources.py
"""
from __future__ import annotations

import re
from pathlib import Path

# Files that make up the kernel. __init__.py is hand-written (it defines the
# public API / __all__) and is intentionally NOT synced.
MODULES = ["router", "permute", "expert_gemm", "fused_moe"]

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SRC_DIR = REPO_ROOT / "triton_kernels" / "moe"
DST_DIR = HERE / "torch-ext" / "moe_dispatch"

# subhadipmitra, 2026-07-19: rewrite package-qualified imports to relative form.
# This is the single transformation that separates the canonical package from the
# Hub-compliant copy; keeping it mechanical means the two never diverge in logic.
ABS_IMPORT = re.compile(r"\bfrom triton_kernels\.moe\.")

BANNER = (
    "# AUTO-GENERATED from triton_kernels/moe/{name}.py by hub/moe-dispatch/sync_sources.py.\n"
    "# Edit the canonical source under triton_kernels/moe/, then re-run the sync script.\n"
)


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for name in MODULES:
        src = (SRC_DIR / f"{name}.py").read_text()
        rewritten = ABS_IMPORT.sub("from .", src)
        if "triton_kernels" in rewritten:
            raise SystemExit(
                f"{name}.py still references 'triton_kernels' after rewrite - "
                "the Hub build must not import the installed package."
            )
        (DST_DIR / f"{name}.py").write_text(BANNER.format(name=name) + rewritten)
        print(f"synced {name}.py")
    print(f"\nWrote {len(MODULES)} modules to {DST_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
