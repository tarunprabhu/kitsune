// RUN: %kitcc -ftapir=none -S -emit-llvm -o - %s | FileCheck %s

#include <kitsune.h>

void bar(int);
void foo(int);

void interleaved() {
  spawn x { bar(1); }

  foo(2);

  spawn y { bar(3); }

  foo(4);

  sync x;

  foo(5);

  sync y;
}

// CHECK:   %[[x:.+]] = call token @llvm.syncregion.start()
// CHECK:   %[[y:.+]] = call token @llvm.syncregion.start()
// CHECK:   detach within %[[x]], label %[[DETACH1:.+]], label %[[CONT1:.+]]
//
// CHECK: [[DETACH1]]:
// CHECK:   call void @bar({{.+}} 1)
// CHECK:   reattach within %[[x]], label %[[CONT1]]
//
// CHECK: [[CONT1]]:
// CHECK:   call void @foo({{.+}} 2)
// CHECK:   detach within %[[y]], label %[[DETACH2:.+]], label %[[CONT2:.+]]
//
// CHECK: [[DETACH2]]:
// CHECK:   call void @bar({{.+}} 3)
// CHECK:   reattach within %[[y]], label %[[CONT2]]
//
// CHECK: [[CONT2]]:
// CHECK:   call void @foo({{.+}} 4)
// CHECK:   sync within %[[x]], label %[[SYNC2:.+]]
//
// CHECK: [[SYNC2]]:
// CHECK:   call void @foo({{.+}} 5)
// CHECK:   sync within %[[y]]
