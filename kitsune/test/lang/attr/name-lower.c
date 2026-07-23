// RUN: %kitcc -std=c23 --tapir=serial -O1 -S -emit-llvm -o - %s %sysroot \
// RUN:     -Xclang -disable-llvm-optzns \
// RUN:     | FileCheck %s
//
// CHECK: !{!"tapir.loop.name", !"roujiamo"}

#include <kitsune.h>

void f(int *a, int n) {
  [[kitsune::name("roujiamo")]]
  forall (int i = 0; i < n; ++i) {
    a[i] = i;
  }
}
