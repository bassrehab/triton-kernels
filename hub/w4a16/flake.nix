{
  description = "W4A16 GEMM - portable 4-bit weight-only Triton kernel (torch-noarch)";

  # Current kernel-builder toolchain (edition 5, genKernelFlakeOutputs). No pinned
  # flake.lock so the build resolves the cached toolchain; see hub/moe-dispatch for why.
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
