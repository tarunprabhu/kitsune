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
// RUN: not %kitxx -### --tapir=cuda --tapir-cuda-arch=gfx90a %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix INVALID
//
// INVALID: error: unsupported NVIDIA GPU architecture 'gfx90a'
//
// -----------------------------------------------------------------------------

// We just need some function to ensure that a tapir target object is created.
void f() {}
