// Check that the default options added to the internal command lines (for -fc1
// and the linker) are as expected.
//
// -----------------------------------------------------------------------------
// RUN: %kitxx -### -ftapir=serial -O2 %s 2>&1 | FileCheck %s
// RUN: %kitxx -### --tapir=serial -O2 %s 2>&1 | FileCheck %s
//
// CHECK: -cc1
// CHECK-SAME: --tapir=serial
// CHECK-SAME: -fstripmine
//
// It is a pain to check for the actual linker executable. There are far too
// many options depending on the platform, so just check the next line for the
// expected linker flags.
//
// CHECK-NEXT: -lkitrt
//
// -----------------------------------------------------------------------------
// Check that the stripmine pass is enabled by default. This checks that the
// the pipeline tuning options object value is set correctly by default.
//
// RUN: %kitxx -mllvm -print-pipeline-passes -O2 -ftapir=serial \
// RUN:      -S -emit-llvm %s | FileCheck %s -check-prefix STRIPMINE-PASS
//
// STRIPMINE-PASS: loop-stripmine
