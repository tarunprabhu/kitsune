; REQUIRES: kitsune-cuda
;
; Check that the launch of a cuda kernel is lowered correctly.
;
; RUN: opt --tapir=cuda -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @launch
; CHECK-NEXT: %[[ARG0:[0-9]+]] = alloca ptr
; CHECK-NEXT: %[[ARG1:[0-9]+]] = alloca i32
; CHECK-NEXT: %[[ARG2:[0-9]+]] = alloca float
; CHECK-NEXT: %[[ARG3:[0-9]+]] = alloca ptr
; CHECK-NEXT: %[[ARRAY:[0-9]+]] = alloca [4 x ptr]
; CHECK-NEXT: store ptr %q, ptr %[[ARG0]]
; CHECK-NEXT: %[[PTR0:[0-9]+]] = getelementptr inbounds [4 x ptr], ptr %[[ARRAY]], i64 0, i64 0
; CHECK-NEXT: store ptr %[[ARG0]], ptr %[[PTR0]]
; CHECK-NEXT: store i32 98, ptr %[[ARG1]]
; CHECK-NEXT: %[[PTR1:[0-9]+]] = getelementptr inbounds [4 x ptr], ptr %[[ARRAY]], i64 0, i64 1
; CHECK-NEXT: store ptr %[[ARG1]], ptr %[[PTR1]]
; CHECK-NEXT: store float %f, ptr %[[ARG2]]
; CHECK-NEXT: %[[PTR2:[0-9]+]] = getelementptr inbounds [4 x ptr], ptr %[[ARRAY]], i64 0, i64 2
; CHECK-NEXT: store ptr %[[ARG2]], ptr %[[PTR2]]
; CHECK-NEXT: store ptr null, ptr %[[ARG3]]
; CHECK-NEXT: %[[PTR3:[0-9]+]] = getelementptr inbounds [4 x ptr], ptr %[[ARRAY]], i64 0, i64 3
; CHECK-NEXT: store ptr %[[ARG3]], ptr %[[PTR3]]
; CHECK-NEXT: call ptr @__kitcuda_launch_kernel(ptr nonnull @fb, ptr nonnull @1, ptr nonnull %[[ARRAY]], i64 %n, i32 0, ptr nonnull @0, ptr %p)
; CHECK-NEXT: ret void

target triple = "x86_64-unknown-linux-gnu"

@fb = external global [23 x i8]
@0 = external global i32
@1 = external global float

define void @launch(ptr %p, ptr dereferenceable(32) %q, i64 %n, float %f) {
  call ptr (i32, ptr, ptr, i64, i32, ptr, ptr, ...) @llvm.kit.async.launch.kernel(i32 2, ptr nonnull @fb, ptr nonnull @1, i64 %n, i32 0, ptr nonnull @0, ptr %p, ptr dereferenceable(32) %q, i32 noundef 98, float %f, ptr null)
  ret void
}
