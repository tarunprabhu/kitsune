// RUN: %kitcc -ftapir=none -S -emit-llvm -o - %s | FileCheck %s

#include <kitsune.h>
#include <stdio.h>

void foo(int);

int main() {

  spawn p1 {
    foo(1);
  }

  spawn p2 {
    foo(2);
  }

  sync p1;
  sync p2;
  return 0;
}

// CHECK: %[[P1:.+]] = call token @llvm.syncregion.start()
// CHECK: %[[P2:.+]] = call token @llvm.syncregion.start()
// CHECK: detach within %[[P1]], label %[[DETACH1:.+]], label %[[CONT1:.+]]
// CHECK: [[DETACH1]]:
// CHECK: call void @foo({{.*}}1)
// CHECK: reattach within %[[P1]], label %[[CONT1]]
// CHECK: [[CONT1]]:
// CHECK: detach within %[[P2]], label %[[DETACH2:.+]], label %[[CONT2:.+]]
// CHECK: [[DETACH2]]:
// CHECK: call void @foo({{.*}}2)
// CHECK: reattach within %[[P2]], label %[[CONT2]]
// CHECK: [[CONT2]]:
// CHECK: sync within %[[P1]]
// CHECK: sync within %[[P2]]
