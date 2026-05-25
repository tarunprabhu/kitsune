// RUN: %kitcc -std=c23 --tapir=nolo -S -emit-llvm -o - %s | FileCheck %s

#include <kitsune.h>

// CHECK-LABEL: @reduce_i8
// CHECK: %[[R:.+]] = alloca i8
// CHECK-NEXT: %[[V:.+]] = load i8, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, ptr %[[R]], i32 1, i8 %[[V]], i8 127,
// CHECK-SAME: ptr @__kitsune_reduce.min.i8)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.min.i8
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: i8 {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load i8, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = call i8 @llvm.smin.i8(i8 %[[CURR]], i8 %[[V]])
// CHECK-NEXT: store i8 %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
void reduce_i8() {
  int8_t r;
  __kitsune_reduce(&r, KIT_MIN, r);
}

// CHECK-LABEL: @reduce_u8
// CHECK: %[[R:.+]] = alloca i8
// CHECK-NEXT: %[[V:.+]] = load i8, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, ptr %[[R]], i32 1, i8 %[[V]], i8 -1,
// CHECK-SAME: ptr @__kitsune_reduce.min.u8)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.min.u8
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: i8 {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load i8, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = call i8 @llvm.umin.i8(i8 %[[CURR]], i8 %[[V]])
// CHECK-NEXT: store i8 %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
void reduce_u8() {
  uint8_t r;
  __kitsune_reduce(&r, KIT_MIN, r);
}

// CHECK-LABEL: @reduce_i16
// CHECK: %[[R:.+]] = alloca i16
// CHECK-NEXT: %[[V:.+]] = load i16, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, ptr %[[R]], i32 2, i16 %[[V]], i16 32767,
// CHECK-SAME: ptr @__kitsune_reduce.min.i16)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.min.i16
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: i16 {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load i16, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = call i16 @llvm.smin.i16(i16 %[[CURR]], i16 %[[V]])
// CHECK-NEXT: store i16 %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
void reduce_i16() {
  int16_t r;
  __kitsune_reduce(&r, KIT_MIN, r);
}

// CHECK-LABEL: @reduce_u16
// CHECK: %[[R:.+]] = alloca i16
// CHECK-NEXT: %[[V:.+]] = load i16, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, ptr %[[R]], i32 2, i16 %[[V]], i16 -1,
// CHECK-SAME: ptr @__kitsune_reduce.min.u16)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.min.u16
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: i16 {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load i16, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = call i16 @llvm.umin.i16(i16 %[[CURR]], i16 %[[V]])
// CHECK-NEXT: store i16 %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
void reduce_u16() {
  uint16_t r;
  __kitsune_reduce(&r, KIT_MIN, r);
}

// CHECK-LABEL: @reduce_i32
// CHECK: %[[R:.+]] = alloca i32
// CHECK-NEXT: %[[V:.+]] = load i32, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, ptr %[[R]], i32 4, i32 %[[V]], i32 2147483647,
// CHECK-SAME: ptr @__kitsune_reduce.min.i32)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.min.i32
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: i32 {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load i32, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = call i32 @llvm.smin.i32(i32 %[[CURR]], i32 %[[V]])
// CHECK-NEXT: store i32 %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
void reduce_i32() {
  int32_t r;
  __kitsune_reduce(&r, KIT_MIN, r);
}

// CHECK-LABEL: @reduce_u32
// CHECK: %[[R:.+]] = alloca i32
// CHECK-NEXT: %[[V:.+]] = load i32, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, ptr %[[R]], i32 4, i32 %[[V]], i32 -1,
// CHECK-SAME: ptr @__kitsune_reduce.min.u32)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.min.u32
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: i32 {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load i32, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = call i32 @llvm.umin.i32(i32 %[[CURR]], i32 %[[V]])
// CHECK-NEXT: store i32 %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
void reduce_u32() {
  uint32_t r;
  __kitsune_reduce(&r, KIT_MIN, r);
}

// CHECK-LABEL: @reduce_i64
// CHECK: %[[R:.+]] = alloca i64
// CHECK-NEXT: %[[V:.+]] = load i64, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, ptr %[[R]], i32 8, i64 %[[V]], i64 9223372036854775807,
// CHECK-SAME: ptr @__kitsune_reduce.min.i64)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.min.i64
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: i64 {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load i64, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = call i64 @llvm.smin.i64(i64 %[[CURR]], i64 %[[V]])
// CHECK-NEXT: store i64 %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
void reduce_i64() {
  int64_t r;
  __kitsune_reduce(&r, KIT_MIN, r);
}

// CHECK-LABEL: @reduce_u64
// CHECK: %[[R:.+]] = alloca i64
// CHECK-NEXT: %[[V:.+]] = load i64, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, ptr %[[R]], i32 8, i64 %[[V]], i64 -1,
// CHECK-SAME: ptr @__kitsune_reduce.min.u64)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.min.u64
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: i64 {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load i64, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = call i64 @llvm.umin.i64(i64 %[[CURR]], i64 %[[V]])
// CHECK-NEXT: store i64 %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
void reduce_u64() {
  uint64_t r;
  __kitsune_reduce(&r, KIT_MIN, r);
}

// CHECK-LABEL: @reduce_f32
// CHECK: %[[R:.+]] = alloca float
// CHECK-NEXT: %[[V:.+]] = load float, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, ptr %[[R]], i32 4, float %[[V]],
// CHECK-SAME: float 0x47EFFFFFE0000000,
// CHECK-SAME: ptr @__kitsune_reduce.min.f32)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.min.f32
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: float {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load float, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = call float @llvm.minimum.f32(float %[[CURR]], float %[[V]])
// CHECK-NEXT: store float %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
void reduce_f32() {
  float r;
  __kitsune_reduce(&r, KIT_MIN, r);
}

// CHECK-LABEL: @reduce_f64
// CHECK: %[[R:.+]] = alloca double
// CHECK-NEXT: %[[V:.+]] = load double, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, ptr %[[R]], i32 8, double %[[V]],
// CHECK-SAME: double 0x7FEFFFFFFFFFFFFF,
// CHECK-SAME: ptr @__kitsune_reduce.min.f64)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.min.f64
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: double {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load double, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = call double @llvm.minimum.f64(double %[[CURR]], double %[[V]])
// CHECK-NEXT: store double %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
void reduce_f64() {
  double r;
  __kitsune_reduce(&r, KIT_MIN, r);
}

// If reducing with the same type more than once, the reducer should not be
// regenerated.
//
// CHECK-NOT: define linkonce_odr void @__kitsune_reduce.min.i64
// CHECK-NOT: define linkonce_odr void @__kitsune_reduce.min.u64
// CHECK-NOT: define linkonce_odr void @__kitsune_reduce.min.f32
void reduce_min_2() {
  int64_t i64;
  uint64_t u64;
  float f32;

  __kitsune_reduce(&i64, KIT_MIN, i64);
  __kitsune_reduce(&u64, KIT_MIN, u64);
  __kitsune_reduce(&f32, KIT_MIN, f32);
}
