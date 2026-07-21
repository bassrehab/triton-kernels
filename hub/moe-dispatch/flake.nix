{
  description = "Fused MoE dispatch - cross-platform Triton kernel (universal build)";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genFlakeOutputs {
      path = ./.;
      rev = self.shortRev or self.dirtyShortRev or self.lastModifiedDate;
      # subhadipmitra, 2026-07-21: skip the post-build get_kernel import check. It
      # needs a full torch/triton/CUDA runtime just to import the kernel, which on a
      # cache-missed toolchain rebuilds LLVM/PyTorch from source (hours). This is a
      # pure-Triton universal kernel with no compiled artifact, and import/correctness
      # were already validated on A100, so the check adds only build time. It is also
      # the documented workaround for triton.autotune kernels in a GPU-less sandbox.
      doGetKernelCheck = false;
    };
}
