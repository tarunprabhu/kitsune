; Check that calls to @kit.reduce.0 intrinsic are lowered correctly.
;
; RUN: opt --passes='kit-lower-reduce-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-NOT: @llvm.kit.reduce.0

declare void @sum(ptr %res, double %v)

; CHECK-LABEL: @fsum
; CHECK-SAME: double %[[V:[^)]+]]
; CHECK-NEXT: %[[RES:.+]] = alloca double
; CHECK-NEXT: call void @sum(ptr %[[RES]], double %[[V]])
define void @fsum(double %v) {
  %res = alloca double
  call void(i32, i32, ptr, i32, double, double, ptr, ...) @llvm.kit.reduce.0(i32 1, i32 6, ptr %res, i32 8, double %v, double 0.0, ptr @sum)
  ret void
}

declare void @mul(ptr %res, i32 %v)

; CHECK-LABEL: @fmul
; CHECK-SAME: i32 %[[V:[^)]+]]
; CHECK-NEXT: %[[RES:.+]] = alloca i32
; CHECK-NEXT: call void @mul(ptr %[[RES]], i32 %[[V]])
define void @fmul(i32 %v) {
  %res = alloca i32
  call void(i32, i32, ptr, i32, i32, i32, ptr, ...) @llvm.kit.reduce.0(i32 1, i32 7, ptr %res, i32 4, i32 %v, i32 1, ptr @mul)
  ret void
}

declare void @custom(ptr %res, ptr %obj, ptr %class, i64 %more)

; CHECK-LABEL: @fcustom
; CHECK-SAME: ptr %[[RES:[^,]+]]
; CHECK-SAME: ptr %[[OBJ:[^,]+]]
; CHECK-SAME: ptr %[[CLASS:[^,]+]]
; CHECK-SAME: i64 %[[MORE:[^)]+]]
; CHECK-NEXT: call void @custom(ptr %[[RES]], ptr %[[OBJ]], ptr %[[CLASS]], i64 %[[MORE]])
define void @fcustom(ptr %res, ptr %obj, ptr %class, i64 %more) {
  call void (i32, i32, ptr, i32, ptr, ptr, ptr, ...) @llvm.kit.reduce.0(i32 1, i32 0, ptr %res, i32 64, ptr %obj, ptr null, ptr @custom, ptr %class, i64 %more)
  ret void
}

; CHECK-LABEL: @ext
; CHECK-SAME: double %[[V:[^,]+]]
; CHECK-SAME: ptr %[[REDUCER:[^)]+]]
; CHECK-NEXT: %[[RES:.+]] = alloca double
; CHECK-NEXT: call void %[[REDUCER]](ptr %[[RES]], double %[[V]])
define void @ext(double %v, ptr %reducer) {
  %res = alloca double
  call void(i32, i32, ptr, i32, double, double, ptr, ...) @llvm.kit.reduce.0(i32 1, i32 0, ptr %res, i32 8, double %v, double 0.0, ptr %reducer)
  ret void
}

; CHECK-LABEL: @ext_vararg
; CHECK-SAME: float %[[V:[^,]+]]
; CHECK-SAME: ptr %[[REDUCER:[^,]+]]
; CHECK-SAME: ptr %[[CLASS:[^,]+]]
; CHECK-SAME: i8 %[[MORE:[^)]+]]
; CHECK-NEXT: %[[RES:.+]] = alloca float
; CHECK-NEXT: call void %[[REDUCER]](ptr %[[RES]], float %[[V]], ptr %[[CLASS]], i8 %[[MORE]])
define void @ext_vararg(float %v, ptr %reducer, ptr %class, i8 %more) {
  %res = alloca float
  call void(i32, i32, ptr, i32, float, float, ptr, ...) @llvm.kit.reduce.0(i32 1, i32 0, ptr %res, i32 8, float %v, float 0.0, ptr %reducer, ptr %class, i8 %more)
  ret void
}
