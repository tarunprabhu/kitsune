// Empty spawn blocks are ok.
//
// RUN: %kitcc --tapir=nolo -S -emit-llvm -o - %s %sysroot | FileCheck %s

#include <kitsune.h>

// CHECK: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
// CHECK: detach within %[[SYNCREG]], label %[[DETACH:.+]], label %[[CONT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT: [[DETACH]]:
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONT]]
// CHECK-EMPTY:
// CHECK-NEXT: [[CONT]]:
// CHECK-NEXT: sync within %[[SYNCREG]]
int main() {
  spawn s{}
  sync s;
}
