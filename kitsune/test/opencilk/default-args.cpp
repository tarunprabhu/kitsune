// -----------------------------------------------------------------------------
// RUN: %kitxx -### -ftapir=opencilk -O2 %s 2>&1 | FileCheck %s
// RUN: %kitxx -### --tapir=opencilk -O2 %s 2>&1 | FileCheck %s
//
// CHECK: -cc1
// CHECK-SAME: --tapir=opencilk
// CHECK-SAME: --tapir-opencilk-abi-bc
// CHECK-SAME: -fstripmine
//
// It is a pain to check for the actual linker executable. There are far too
// many options depending on the platform, so just check the next line for the
// expected linker flags.
//
// CHECK-NEXT: -lopencilk
// CHECK-SAME: -lkitrt
