// -----------------------------------------------------------------------------

// RUN: not %clang -### -ftapir-hip-arch=gfx906 -ftapir=hip %s 2>&1    \
// RUN:     | FileCheck %s -check-prefix FRONTEND

// FRONTEND: option '-ftapir-hip-arch=' must be used with a Kitsune frontend

// -----------------------------------------------------------------------------

// The -ftapir-hip-arch is not used if the -ftapir options is not hip, or if
// the -ftapir flag was not provided.
// RUN: %kitxx -### -ftapir-hip-arch=gfx906 %s 2>&1  \
// RUN:     | FileCheck %s -check-prefix UNUSED
// RUN: %kitxx -### -ftapir-hip-arch=gfx906 -ftapir=serial %s 2>&1  \
// RUN:     | FileCheck %s -check-prefix UNUSED

// UNUSED-NOT: '-ftapir-cuda-arch=sm_72'

// -----------------------------------------------------------------------------

// If the tapir target is hip, the architecture should be passed on to cc1
// RUN: %kitxx -### -ftapir-hip-arch=gfx906 -ftapir=hip %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix USED

// USED: -ftapir-hip-arch=gfx906

// -----------------------------------------------------------------------------

// Make sure that the architecture makes it to hipabi.
// RUN: %kitxx -ftapir=hip -ftapir-hip-arch=gfx90c -mllvm -hipabi-### \
// RUN:     -S -emit-llvm -O2  %s 2>&1 | FileCheck %s -check-prefix LOWERED

// LOWERED: lld
// LOWERED: -plugin-opt=mcpu=gfx90c

// -----------------------------------------------------------------------------

// RUN: not %kitxx -### -ftapir=hip -ftapir-hip-arch=sm_80 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix INVALID

// INVALID: error: invalid value 'sm_80' in '-ftapir-hip-arch=sm_80'

// -----------------------------------------------------------------------------

// We need some code there that contains a forall that will force HipABI to
// run.
#include <kitsune.h>

void f(unsigned *buf, unsigned n) {
  // clang-format off
  forall(int i = 0; i < n; ++i) {
    buf[i] = i;
  }
  // clang-format on
}
