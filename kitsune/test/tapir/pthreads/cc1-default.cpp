// -----------------------------------------------------------------------------
// Check that the default options added to the internal command lines (for -cc1
// and the linker) are as expected.
//
// RUN: %kitxx -### --tapir=pthreads -O2 %s 2>&1 | FileCheck %s
//
// CHECK: -cc1
// CHECK-SAME: --tapir=pthreads
//
// CHECK-NOT: -fstripmine
//
// The next line is expected to be the linker invocation. Since it is difficult
// to reliably check the name of the linker executable, just check for the
// expected linker flags.
//
// CHECK-NEXT: -lkitrt
//
// -----------------------------------------------------------------------------
// Check that the stripmine pass is disabled by default. This checks that the
// pipeline tuning options object is setup correctly.
//
// RUN: %kitxx -mllvm -print-pipeline-passes -O2 --tapir=pthreads \
// RUN:     -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE-PASS
//
// STRIPMINE-PASS-NOT: loop-stripmine
