// -----------------------------------------------------------------------------

// RUN: not %clang -### -ftapir-cuda-arch=sm_72 -ftapir=cuda %s 2>&1    \
// RUN:     | FileCheck %s -check-prefix FRONTEND

// FRONTEND: option '-ftapir-cuda-arch=' must be used with a Kitsune frontend

// -----------------------------------------------------------------------------

// The -ftapir-cuda-arch is not used if the -ftapir options is not cuda, or if
// the -ftapir flag was not provided.
// RUN: %kitxx -### -ftapir-cuda-arch=sm_72 %s 2>&1  \
// RUN:     | FileCheck %s -check-prefix UNUSED
// RUN: %kitxx -### -ftapir-cuda-arch=sm_72 -ftapir=serial %s 2>&1  \
// RUN:     | FileCheck %s -check-prefix UNUSED

// UNUSED-NOT: '-ftapir-cuda-arch=sm_72'

// -----------------------------------------------------------------------------

// If the tapir target is cuda, the architecture should be passed on to cc1
// RUN: %kitxx -### -ftapir-cuda-arch=sm_72 -ftapir=cuda %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix USED

// USED: -ftapir-cuda-arch=sm_72

// -----------------------------------------------------------------------------

// Make sure that the architecture makes it to cuabi.
// RUN: %kitxx -ftapir=cuda -ftapir-cuda-arch=sm_86 -mllvm -cuabi-### \
// RUN:     -S -emit-llvm -O2  %s 2>&1 | FileCheck %s -check-prefix LOWERED

// LOWERED: ptxas
// LOWERED-SAME: --gpu-name
// LOWERED-SAME: sm_86

// -----------------------------------------------------------------------------

// RUN: not %kitxx -### -ftapir=cuda -ftapir-cuda-arch=gfx90a %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix INVALID

// INVALID: error: invalid value 'gfx90a' in '-ftapir-cuda-arch=gfx90a'

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
