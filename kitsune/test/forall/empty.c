// Check that empty forall's are lowered correctly. It is ok for them to be
// empty, though, with optimizations enabled, they are likely to be DCE'ed
//
// RUN: %kitcc -ftapir=none -S -emit-llvm -o - %s | FileCheck %s

#include <kitsune.h>

void loop() {
  forall(int i = 0; i < 10; i++) {
  }
}

// CHECK: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
// CHECK: detach within %[[SYNCREG:.+]], label %[[BODY:.+]], label %[[INC:.+]]
// CHECK: [[BODY]]
// CHECK store
// CHECK: br label %[[REATTACH:.+]]
// CHECK: [[REATTACH]]:
// CHECK: reattach within %[[SYNCREG]], label %[[INC]]
// CHECK: [[INC]]:
// CHECK: br label {{.+}}, !llvm.loop
// CHECK: [[SYNC:.+]]:
// CHECK: sync within %[[SYNCREG]]
