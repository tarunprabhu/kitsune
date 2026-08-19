// RUN: %kitxx --tapir=nolo -S -emit-llvm -o - %s | FileCheck %s

#include <kitsune.h>

// CHECK-LABEL: @reduce_i8
// CHECK: %[[R:.+]] = alloca i8
// CHECK-NEXT: %[[V:.+]] = load i8, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, i32 2, ptr %[[R]], i32 1, i8 %[[V]], i8 0,
// CHECK-SAME: ptr @__kitsune_reduce.or.u8)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.or.u8
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: i8 {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load i8, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = or i8 %[[CURR]], %[[V]]
// CHECK-NEXT: store i8 %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
extern "C" void reduce_i8() {
  int8_t r;
  __kitsune_reduce(&r, KIT_OR, r);
}

// CHECK-LABEL: @reduce_u8
// CHECK: %[[R:.+]] = alloca i8
// CHECK-NEXT: %[[V:.+]] = load i8, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, i32 2, ptr %[[R]], i32 1, i8 %[[V]], i8 0,
// CHECK-SAME: ptr @__kitsune_reduce.or.u8)
// CHECK-NEXT: ret void
extern "C" void reduce_u8() {
  uint8_t r;
  __kitsune_reduce(&r, KIT_OR, r);
}

// CHECK-LABEL: @reduce_i16
// CHECK: %[[R:.+]] = alloca i16
// CHECK-NEXT: %[[V:.+]] = load i16, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, i32 2, ptr %[[R]], i32 2, i16 %[[V]], i16 0,
// CHECK-SAME: ptr @__kitsune_reduce.or.u16)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.or.u16
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: i16 {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load i16, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = or i16 %[[CURR]], %[[V]]
// CHECK-NEXT: store i16 %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
extern "C" void reduce_i16() {
  int16_t r;
  __kitsune_reduce(&r, KIT_OR, r);
}

// CHECK-LABEL: @reduce_u16
// CHECK: %[[R:.+]] = alloca i16
// CHECK-NEXT: %[[V:.+]] = load i16, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, i32 2, ptr %[[R]], i32 2, i16 %[[V]], i16 0,
// CHECK-SAME: ptr @__kitsune_reduce.or.u16)
// CHECK-NEXT: ret void
extern "C" void reduce_u16() {
  uint16_t r;
  __kitsune_reduce(&r, KIT_OR, r);
}

// CHECK-LABEL: @reduce_i32
// CHECK: %[[R:.+]] = alloca i32
// CHECK-NEXT: %[[V:.+]] = load i32, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, i32 2, ptr %[[R]], i32 4, i32 %[[V]], i32 0,
// CHECK-SAME: ptr @__kitsune_reduce.or.u32)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.or.u32
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: i32 {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load i32, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = or i32 %[[CURR]], %[[V]]
// CHECK-NEXT: store i32 %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
extern "C" void reduce_i32() {
  int32_t r;
  __kitsune_reduce(&r, KIT_OR, r);
}

// CHECK-LABEL: @reduce_u32
// CHECK: %[[R:.+]] = alloca i32
// CHECK-NEXT: %[[V:.+]] = load i32, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, i32 2, ptr %[[R]], i32 4, i32 %[[V]], i32 0,
// CHECK-SAME: ptr @__kitsune_reduce.or.u32)
// CHECK-NEXT: ret void
extern "C" void reduce_u32() {
  uint32_t r;
  __kitsune_reduce(&r, KIT_OR, r);
}

// CHECK-LABEL: @reduce_i64
// CHECK: %[[R:.+]] = alloca i64
// CHECK-NEXT: %[[V:.+]] = load i64, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, i32 2, ptr %[[R]], i32 8, i64 %[[V]], i64 0,
// CHECK-SAME: ptr @__kitsune_reduce.or.u64)
//
// CHECK-LABEL: define linkonce_odr void @__kitsune_reduce.or.u64
// CHECK-SAME: ptr %[[PTR:[^,]+]]
// CHECK-SAME: i64 {{.*}}%[[V:[^)]+]]
// CHECK-NEXT: %[[CURR:.+]] = load i64, ptr %[[PTR]]
// CHECK-NEXT: %[[UPD:.+]] = or i64 %[[CURR]], %[[V]]
// CHECK-NEXT: store i64 %[[UPD]], ptr %[[PTR]]
// CHECK-NEXT: ret void
extern "C" void reduce_i64() {
  int64_t r;
  __kitsune_reduce(&r, KIT_OR, r);
}

// CHECK-LABEL: @reduce_u64
// CHECK: %[[R:.+]] = alloca i64
// CHECK-NEXT: %[[V:.+]] = load i64, ptr %[[R]]
// CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{[^(]+}}
// CHECK-SAME: (i32 0, i32 2, ptr %[[R]], i32 8, i64 %[[V]], i64 0,
// CHECK-SAME: ptr @__kitsune_reduce.or.u64)
// CHECK-NEXT: ret void
extern "C" void reduce_u64() {
  uint64_t r;
  __kitsune_reduce(&r, KIT_OR, r);
}

// If reducing with the same type more than once, the reducer should not be
// regenerated.
//
// CHECK-NOT: define linkonce_odr void @__kitsune_reduce.or.u32
// CHECK-NOT: define linkonce_odr void @__kitsune_reduce.or.u64
void reduce_or_2() {
  int32_t i32;
  int64_t i64;
  uint64_t u64;

  __kitsune_reduce(&i32, KIT_OR, i32);
  __kitsune_reduce(&i64, KIT_OR, i64);
  __kitsune_reduce(&u64, KIT_OR, u64);
}
