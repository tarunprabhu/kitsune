// -----------------------------------------------------------------------------
// Check that the default options added to the internal command lines (for -cc1
// and the linker) are as expected.
//
// RUN: %kitxx -### --tapir=hip -O2 %s 2>&1 | FileCheck %s
//
// -cc1 must always get the GPU architecture, bitcode files, features, the full
// path to LLD and the values of sramecc and xnack.
//
// CHECK: -cc1
// CHECK-SAME: --tapir=hip
// CHECK-SAME: --tapir-hip-arch={{[^"]+}}"
// CHECK-SAME: --tapir-hip-sramecc=on
// CHECK-SAME: --tapir-hip-xnack=on
// CHECK-SAME: --tapir-hip-features={{[^"]*}}"
// CHECK-SAME: --tapir-hip-runtime-bcs={{[^"]+}}"
// CHECK-SAME: --tapir-lld={{[^"]+}}"
// CHECK-SAME: --tapir-gpu-prefetch
//
// CHECK-NOT: -fstripmine
//
// The next line is expected to be the linker invocation. Since it is difficult
// to reliably check the name of the linker executable, just check for the
// expected linker flags.
//
// CHECK-NEXT: -lkitrt
// CHECK-SAME: -lamdhip64
//
// -----------------------------------------------------------------------------
// Check that the stripmine pass is disabled by default. This checks that the
// pipeline tuning options object value is set correctly by default.
//
// RUN: %kitxx -mllvm -print-pipeline-passes -O2 --tapir=hip \
// RUN:     -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE-PASS
//
// STRIPMINE-PASS-NOT: loop-stripmine
//
// -----------------------------------------------------------------------------
