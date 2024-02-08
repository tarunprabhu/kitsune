// `continue` statements are allowed in a forall loop.
//
// RUN: %kitxx --tapir=nolo -S -emit-llvm -o - %s %sysroot | FileCheck %s

#include <kitsune.h>

// CHECK: detach within %[[SYNCREG:.+]], label {{.+}}, label {{.+}}
// CHECK: %[[CMP:.+]] = icmp eq {{.+}}, 4
// CHECK: br i1 %[[CMP]], label %[[THEN:.+]], label %[[ELSE:.+]]
// CHECK: [[THEN]]:
// CHECK-NEXT: br label %[[REATTACH:.+]]
// CHECK: [[ELSE]]:
// CHECK: store
// CHECK-NEXT: br label %[[REATTACH]]
// CHECK: [[REATTACH]]:
// CHECK-NEXT: reattach within %[[SYNCREG]]
void loop(int *a, int n) {
  forall(int i = 0; i < n; i++) {
    if (i == 4)
      continue;
    a[i] = i;
  }
}
