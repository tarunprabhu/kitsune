// The -ftapir-hip-arch option is currently unused. This test will have to be
// updated when we handle it.

// RUN: not %clang -### -ftapir-hip-arch=gfx90a -ftapir=hip %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix FRONTEND

// FRONTEND: option '-ftapir-hip-arch=' must be used with a Kitsune frontend

// RUN: %kitxx -### -ftapir-hip-arch=sm_80 -ftapir=hip %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix UNUSED

// UNUSED: argument unused during compilation: '-ftapir-hip-arch=gfx90a'
