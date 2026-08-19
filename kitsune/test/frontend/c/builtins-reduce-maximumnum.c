// RUN: %kitcc -std=c23 --tapir=nolo -S -emit-llvm -o - %s | FileCheck %s

#include <kitsune.h>

// CHECK-LABEL: @reduce_f32
// CHECK: %[[R:.+]] = alloca float
// CHECK-NEXT: %[[V:.+]] = load float, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, i32 18, ptr %[[R]], i32 4, float %[[V]],
// CHECK-SAME: float 0x3810000000000000,
// CHECK-SAME: ptr @__kitsune_reduce.maximumnum.f32)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.maximumnum.f32
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: float {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load float, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = call float @llvm.maximumnum.f32(float %[[CURR]], float %[[V]])
// CHECK-NEXT: store float %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
void reduce_f32() {
  float r;
  __kitsune_reduce(&r, KIT_MAXIMUM_NUM, r);
}

// CHECK-LABEL: @reduce_f64
// CHECK: %[[R:.+]] = alloca double
// CHECK-NEXT: %[[V:.+]] = load double, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, i32 18, ptr %[[R]], i32 8, double %[[V]],
// CHECK-SAME: double 0x10000000000000,
// CHECK-SAME: ptr @__kitsune_reduce.maximumnum.f64)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.maximumnum.f64
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: double {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load double, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = call double @llvm.maximumnum.f64(double %[[CURR]], double %[[V]])
// CHECK-NEXT: store double %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
void reduce_f64() {
  double r;
  __kitsune_reduce(&r, KIT_MAXIMUM_NUM, r);
}

// If reducing with the same type more than once, the reducer should not be
// regenerated.
//
// CHECK-NOT: define linkonce_odr void @__kitsune_reduce.maximumnum.f32
// CHECK-NOT: define linkonce_odr void @__kitsune_reduce.maximumnum.f64
void reduce_max_2() {
  float f32;
  double f64;

  __kitsune_reduce(&f32, KIT_MAXIMUM_NUM, f32);
  __kitsune_reduce(&f64, KIT_MAXIMUM_NUM, f64);
}
