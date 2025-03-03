// -----------------------------------------------------------------------------
//
// RUN: not %clang -### --tapir-hip-arch=gfx906 --tapir=hip %s 2>&1    \
// RUN:     | FileCheck %s -check-prefix FRONTEND
//
// FRONTEND: option '--tapir-hip-arch=' must be used with a Kitsune frontend
//
// -----------------------------------------------------------------------------
//
// The --tapir-hip-arch is not used if the tapir target is not hip, or if the
// tapir target is not set.
//
// RUN: %kitxx -### --tapir-hip-arch=gfx906 %s 2>&1  \
// RUN:     | FileCheck %s -check-prefix UNUSED
// RUN: %kitxx -### --tapir-hip-arch=gfx906 --tapir=serial %s 2>&1  \
// RUN:     | FileCheck %s -check-prefix UNUSED
//
// UNUSED-NOT: '--tapir-hip-arch=gfx906'
//
// -----------------------------------------------------------------------------
//
// If the tapir target is hip, the architecture should be passed on to cc1
// RUN: %kitxx -### --tapir-hip-arch=gfx906 --tapir=hip %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix USED
//
// USED: -ftapir-hip-arch=gfx906
//
// -----------------------------------------------------------------------------
//
// Make sure that the architecture makes it to hipabi.
// RUN: %kitxx --tapir=hip --tapir-hip-arch=gfx90c -mllvm -hipabi-### \
// RUN:     -S -emit-llvm -O2  %s 2>&1 | FileCheck %s -check-prefix LOWERED
//
// LOWERED: lld
// LOWERED: -plugin-opt=mcpu=gfx90c
//
// -----------------------------------------------------------------------------
//
// RUN: not %kitxx -### --tapir=hip --tapir-hip-arch=sm_80 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix INVALID
//
// INVALID: error: invalid value 'sm_80' in '--tapir-hip-arch=sm_80'
//
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
