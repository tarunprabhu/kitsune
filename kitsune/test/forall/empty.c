// RUN: %clang -ftapir=none -S -emit-llvm -o - %s | FileCheck %s

#include <kitsune.h>

void loop(int* a) {
  // Empty forall's are ok.
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
