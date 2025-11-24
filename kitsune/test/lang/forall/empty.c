// Check that empty forall's are lowered correctly. It is ok for them to be
// empty, though, with optimizations enabled, they are likely to be DCE'ed
//
// RUN: %kitcc --tapir=nolo -S -emit-llvm -o - %s %sysroot | FileCheck %s
//
// CHECK: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
// CHECK: br label %[[COND:.+]]
// CHECK: [[COND]]:
// CHECK: br {{.+}}, label %[[DETACH:.+]], label %[[SYNC:.+]]
// CHECK: [[DETACH]]:
// CHECK: detach within %[[SYNCREG:.+]], label %[[BODY:.+]], label %[[INC:.+]]
// CHECK: [[BODY]]
// CHECK: br label %[[REATTACH:.+]]
// CHECK: [[REATTACH]]:
// CHECK: reattach within %[[SYNCREG]], label %[[INC]]
// CHECK: [[INC]]:
// CHECK: br label %[[COND]]
// CHECK: [[SYNC]]:
// CHECK: sync within %[[SYNCREG]]

#include <kitsune.h>

void loop() {
  forall(int i = 0; i < 10; i++) {
  }
}

