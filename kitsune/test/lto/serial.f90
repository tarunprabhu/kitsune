! RUN: %kitfc -### -ftapir=serial -O2 -flto %s 2>&1 | FileCheck %s

! CHECK: -dynamic-linker
! CHECK-SAME: -plugin
! CHECK-SAME: LLVMgold.so
! CHECK-SAME: --plugin-opt=tapir=serial
