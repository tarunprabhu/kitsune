// -----------------------------------------------------------------------------
// RUN: %kitxx -### -ftapir=hip -O2 %s 2>&1 | FileCheck %s
// RUN: %kitxx -### --tapir=hip -O2 %s 2>&1 | FileCheck %s
//
// CHECK: -cc1
// CHECK-SAME: --tapir=hip
//
// Strip-mining is disabled by default on GPU tapir targets.
//
// CHECK-NOT: -fstripmine
//
// It is a pain to check for the actual linker executable. There are far too
// many options depending on the platform, so just check the next line for the
// expected linker flags.
//
// CHECK-NEXT: -lkitrt
// CHECK-SAME: -lamdhip64
//
// -----------------------------------------------------------------------------
// Check that the stripmine pass is enabled/disabled correctly.
//
// RUN: %kitxx -mllvm -print-pipeline-passes -O2 -fstripmine -ftapir=hip \
// RUN:      -S -emit-llvm %s | FileCheck %s -check-prefix STRIPMINE-PASS
//
// STRIPMINE-PASS: loop-stripmine
