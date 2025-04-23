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
// RUN: not %kitxx -### --tapir=hip --tapir-hip-arch=sm_80 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix INVALID
//
// INVALID: error: unsupported AMD GPU architecture 'sm_80'
//
// -----------------------------------------------------------------------------

// We just need some function to ensure that a tapir target object is created.
void f() {}
