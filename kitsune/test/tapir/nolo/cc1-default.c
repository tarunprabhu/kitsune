// -----------------------------------------------------------------------------
// Check that the default options added to the internal command lines (for -cc1
// and the linker) are as expected.
//
// RUN: %kitcc -### --tapir=nolo -O2 %s 2>&1 | FileCheck %s
//
// CHECK: -cc1
// CHECK-SAME: --tapir=nolo
//
// CHECK-NOT: -fstripmine
//
// This pass adds nothing to the linker, so there is nothing to check for. But
// we do check for the absence of libkitrt.
//
// CHECK-NOT: -lkitrt
//
// -----------------------------------------------------------------------------
// Check that the stripmine pass is disabled by default. This checks that the
// pipeline tuning options object is setup correctly.
//
// RUN: %kitcc -mllvm -print-pipeline-passes -O2 --tapir=nolo \
// RUN:     -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE-PASS
//
// STRIPMINE-PASS-NOT: loop-stripmine
//
// -----------------------------------------------------------------------------
