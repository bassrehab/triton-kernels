{
  description = "Fused MoE dispatch - cross-platform Triton kernel (torch-noarch)";

  # subhadipmitra, 2026-07-21: the builder now lives in huggingface/kernels and uses
  # genKernelFlakeOutputs (edition 5). Schema mirrors the current relu-triton example.
  # Not pinning a flake.lock so the CURRENT (cached) toolchain is used; the previous
  # pinned old release cache-missed and rebuilt the whole torch/CUDA/LLVM closure from
  # source (hours).
  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
