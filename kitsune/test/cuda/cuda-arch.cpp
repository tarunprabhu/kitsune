// -----------------------------------------------------------------------------
//
// RUN: not %clang -### --tapir-cuda-arch=sm_72 --tapir=cuda %s 2>&1    \
// RUN:     | FileCheck %s -check-prefix FRONTEND
//
// FRONTEND: option '--tapir-cuda-arch=' must be used with a Kitsune frontend
//
// -----------------------------------------------------------------------------
//
// The --tapir-cuda-arch is not used if the tapir target is not cuda, or if the
// tapir target is not set.
//
// RUN: %kitxx -### --tapir-cuda-arch=sm_72 %s 2>&1  \
// RUN:     | FileCheck %s -check-prefix UNUSED
// RUN: %kitxx -### --tapir-cuda-arch=sm_72 -ftapir=serial %s 2>&1  \
// RUN:     | FileCheck %s -check-prefix UNUSED
//
// UNUSED-NOT: '--tapir-cuda-arch=sm_72'
//
// -----------------------------------------------------------------------------
//
// If the tapir target is cuda, the architecture should be passed on to cc1
//
// RUN: %kitxx -### --tapir-cuda-arch=sm_72 -ftapir=cuda %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix USED
//
// USED: --tapir-cuda-arch=sm_72
//
// -----------------------------------------------------------------------------
//
// Make sure that the architecture makes it to cuabi.
//
// RUN: %kitxx --tapir-verbose --tapir=cuda --tapir-cuda-arch=sm_86 \
// RUN:     -S -emit-llvm -O2 -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix LOWERED
//
// LOWERED: /ptxas
// LOWERED-SAME: --gpu-name
// LOWERED-SAME: sm_86
//
// -----------------------------------------------------------------------------
//
// RUN: not %kitxx -### --tapir=cuda --tapir-cuda-arch=gfx90a %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix INVALID
//
// INVALID: error: unsupported NVIDIA GPU architecture 'gfx90a'
//
// -----------------------------------------------------------------------------

// We need some code there that contains a forall that will force CudaABI to
// run.
#include <kitsune.h>

void f(unsigned *buf, unsigned n) {
  // clang-format off
  forall(int i = 0; i < n; ++i) {
    buf[i] = i;
  }
  // clang-format on
}
