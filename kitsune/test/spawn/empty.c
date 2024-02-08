// RUN: %clang -ftapir=none -S -emit-llvm -o - %s | FileCheck %s

#include <kitsune.h>

int main() {
  spawn s{}
  sync s;
}

// CHECK: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
// CHECK: detach within %[[SYNCREG]], label %[[DETACH:.+]], label %[[CONT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT: [[DETACH]]:
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONT]]
// CHECK-EMPTY:
// CHECK-NEXT: [[CONT]]:
// CHECK-NEXT: sync within %[[SYNCREG]]
