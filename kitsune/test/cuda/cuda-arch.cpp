// The -ftapir-cuda-arch option is currently unused. This test will have to be
// updated when we handle it.

// RUN: not %clang -### -ftapir-cuda-arch=sm_80 -ftapir=cuda %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix FRONTEND

// FRONTEND: option '-ftapir-cuda-arch=' must be used with a Kitsune frontend

// RUN: %kitxx -### -ftapir-cuda-arch=sm_80 -ftapir=cuda %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix UNUSED

// UNUSED: argument unused during compilation: '-ftapir-cuda-arch=sm_80'
