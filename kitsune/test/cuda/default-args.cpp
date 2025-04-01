// Check that the default options added to the internal command lines (for -cc1
// and the linker) are as expected.
//
// -----------------------------------------------------------------------------
// RUN: %kitxx -### -ftapir=cuda -O2 %s 2>&1 | FileCheck %s
// RUN: %kitxx -### --tapir=cuda -O2 %s 2>&1 | FileCheck %s
//
// -cc1 must always get the GPU architecture, virtual architecture and PTX
// version arguments.
//
// CHECK: -cc1
// CHECK-SAME: --tapir=cuda
//
// Stripmining is disabled by default on GPU tapir targets.
//
// CHECK-NOT: -fstripmine
//
// It is a pain to check for the actual linker executable. There are far too
// many options depending on the platform, so just check the next line for the
// expected linker flags.
//
// CHECK-NEXT: -lkitrt
// CHECK-SAME: -lcudart_static
// CHECK-SAME: -lcuda
//
// ----------------------------------------------------------------------------
// Check that the stripmine pass is disabled by default. This checks that the
// the pipeline tuning options object value is set correctly by default.
//
// RUN: %kitxx -mllvm -print-pipeline-passes -O2 -ftapir=cuda \
// RUN:      -S -emit-llvm %s | FileCheck %s -check-prefix STRIPMINE-PASS
//
// STRIPMINE-PASS-NOT: loop-stripmine
