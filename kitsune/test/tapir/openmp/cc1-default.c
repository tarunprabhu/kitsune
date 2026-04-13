// -----------------------------------------------------------------------------
// Check that the default options added to the internal command lines (for -cc1
// and the linker) are as expected.
//
// RUN: %kitcc -### --tapir=openmp -O2 %s 2>&1 | FileCheck %s
//
// CHECK: -cc1
// CHECK-SAME: --tapir=openmp
// CHECK-SAME: -fstripmine
//
// It is a pain to check for the actual linker executable. There are far too
// many options depending on the platform, so just check the next line for the
// expected linker flags.
//
// CHECK-NEXT: -lomp
// CHECK-SAME: -lkitrt
//
// -----------------------------------------------------------------------------
// Check that the stripmine pass is enabled by default.
//
// RUN: %kitcc -mllvm -print-pipeline-passes -O2 --tapir=openmp \
// RUN:      -S -emit-llvm %s | FileCheck %s -check-prefix STRIPMINE-PASS
//
// STRIPMINE-PASS: loop-stripmine
//
// -----------------------------------------------------------------------------
