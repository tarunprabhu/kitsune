// RUN: %kitcc -ftapir=none -S -emit-llvm -o - %s | FileCheck %s

#include <kitsune.h>

void f() {
  spawn s{}
}

// A spawn without a sync is perfectly legal. Probably not wise, but it is
// legal.
// Should this actually be an error?
//
// CHECK: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
// CHECK-NEXT: detach within %[[SYNCREG]], label %[[DETACH:.+]], label %[[CONT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT: [[DETACH]]:
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONT]]
// CHECK-EMPTY:
// CHECK-NEXT: [[CONT]]:
// CHECK-NEXT: ret void
