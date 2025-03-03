! REQUIRES: kitsune-opencilk

! RUN: %kitfc -### -ftapir=opencilk -O2 -flto %s 2>&1 | FileCheck %s

! CHECK: -dynamic-linker
! CHECK-SAME: -plugin
! CHECK-SAME: LLVMgold.so
! CHECK-SAME: --plugin-opt=tapir=opencilk
! CHECK-SAME: --plugin-opt=tapir-opencilk-abi-bc={{.+}}/libopencilk-abi.bc
