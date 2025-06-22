// Check that the --tapir-hip-arch option is handled correctly.
//
// -----------------------------------------------------------------------------
//
// RUN: not %kitxx -### --tapir=hip --tapir-hip-arch=sm_80 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix INVALID
//
// INVALID: error: unsupported AMD GPU architecture 'sm_80'
//
// -----------------------------------------------------------------------------
//
// RUN: %kitxx -### --tapir=hip --tapir-hip-arch=gfx906 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix OK
//
// OK: -cc1
// OK-SAME: --tapir-hip-arch=gfx906
