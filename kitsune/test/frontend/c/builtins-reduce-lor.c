// RUN: %kitcc -std=c23 --tapir=nolo -S -emit-llvm -o - %s | FileCheck %s

#include <kitsune.h>

// CHECK-LABEL: @reduce_lor
// CHECK-NEXT: [[ENTRY:.+]]:
// CHECK-NEXT: %[[B:.+]] = alloca i8
//
// CHECK-NEXT: %[[VB8:.+]] = load i8, ptr %[[B]]
// CHECK-NEXT: %[[VB:.+]] = trunc i8 %[[VB8]] to i1
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, ptr %[[B]], i32 1, i1 %[[VB]], i1 false,
// CHECK-SAME: ptr @__kitsune_reduce.lor.i1)
//
// CHECK-NEXT: ret void
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.lor.i1
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: i1 {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load i1, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = or i1 %[[CURR]], %[[V]]
// CHECK-NEXT: store i1 %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void

void reduce_lor() {
  bool b;

  __kitsune_reduce(&b, KIT_LOR, b);
}

// If reducing with the same type more than once, the reducer should not be
// regenerated.
//
// CHECK-NOT: define linkonce_odr void @__kitsune_reduce.lor.i1
void reduce_lor_2() {
  bool b;

  __kitsune_reduce(&b, KIT_LOR, b);
}
