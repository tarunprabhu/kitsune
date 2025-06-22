// The default FP contract value is different from clang's defaults. Check that
// the defaults have not changed when running without a Kitsune frontend.
//
// RUN: %clang -### -x cuda -nocudalib -nocudainc %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix DEFAULT-CUDA
//
// When compiling for cuda, the host and device code are compiled separately
// The device code compilation (with a "primary" triple indicating nvptx) does
// not contain an ffp-contract entry - presumably because it is set to "fast"
// internally. On the host (with an "auxiliary" triple indicating nvptx), the
// fp-contract value remains set to the default of "ON".
//
// DEFAULT-CUDA: "-triple" "nvptx{{.*}}-nvidia-cuda"
// DEFAULT-CUDA-NOT: -ffp-contract
// DEFAULT-CUDA: "-aux-triple" "nvptx{{.*}}-nvidia-cuda"
// DEFAULT-CUDA-SAME: "-ffp-contract=on"
