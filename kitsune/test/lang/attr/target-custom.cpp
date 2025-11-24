// Unlike the other tapir targets, "custom" is not a valid argument for the
// tapir::target attribute
//
// RUN: not %kitxx -O1 --tapir=serial -S -emit-llvm -o - %s %sysroot 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: value of tapir::target attribute cannot be 'custom'

#include <kitsune.h>

void f(int *a, int scale, size_t n) {
  // clang-format off
  [[tapir::target("custom")]]
  forall (size_t i = 0; i < n; ++i) {
    a[i] *= scale;
  }
  // clang-format on
}
