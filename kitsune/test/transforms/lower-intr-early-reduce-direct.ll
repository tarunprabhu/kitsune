; Check that the kit.reduce intrinsic with the `Direct` strategy is lowered
; correctly by the kit-lower-intrinsics-early-pass. This will only lower calls
; with a CPU-centric tapir target.
;
; RUN: opt -passes=kit-lower-intrinsics-early -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-NEXT: atomicrmw and ptr %result, i32 %i monotonic
; CHECK-NEXT: atomicrmw or ptr %result, i32 %i monotonic
; CHECK-NEXT: atomicrmw xor ptr %result, i32 %i monotonic
; CHECK-NEXT: atomicrmw add ptr %result, i32 %i monotonic
; CHECK-NEXT: atomicrmw fadd ptr %result, double %d monotonic
; CHECK-NEXT: atomicrmw fmax ptr %result, double %d monotonic
; CHECK-NEXT: atomicrmw fmaximum ptr %result, double %d monotonic
; CHECK-NEXT: atomicrmw fmaximumnum ptr %result, double %d monotonic
; CHECK-NEXT: atomicrmw fmin ptr %result, double %d monotonic
; CHECK-NEXT: atomicrmw fminimum ptr %result, double %d monotonic
; CHECK-NEXT: atomicrmw fminimumnum ptr %result, double %d monotonic
; CHECK-NEXT: atomicrmw max ptr %result, i32 %i monotonic
; CHECK-NEXT: atomicrmw min ptr %result, i32 %i monotonic
; CHECK-NEXT: atomicrmw umax ptr %result, i32 %i monotonic
; CHECK-NEXT: atomicrmw umin ptr %result, i32 %i monotonic
; CHECK-NEXT: ret void
define void @f(ptr %result, i32 %i, double %d) {
  call void(i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce(i32 512, i32 1, ptr %result, i32 %i, i32 2, i32 4, i32 0, ptr @and)
  call void(i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce(i32 512, i32 2, ptr %result, i32 %i, i32 2, i32 4, i32 0, ptr @or)
  call void(i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce(i32 512, i32 3, ptr %result, i32 %i, i32 2, i32 4, i32 0, ptr @xor)
  call void(i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce(i32 512, i32 5, ptr %result, i32 %i, i32 2, i32 4, i32 0, ptr @add)
  call void(i32, i32, ptr, double, i32, i32, double, ptr, ...) @llvm.kit.reduce(i32 512, i32 6, ptr %result, double %d, i32 2, i32 8, double 0.000000E+00, ptr @fadd)
  call void(i32, i32, ptr, double, i32, i32, double, ptr, ...) @llvm.kit.reduce(i32 512, i32 16, ptr %result, double %d, i32 2, i32 8, double 0.000000E+00, ptr @fmax)
  call void(i32, i32, ptr, double, i32, i32, double, ptr, ...) @llvm.kit.reduce(i32 512, i32 17, ptr %result, double %d, i32 2, i32 8, double 0.000000E+00, ptr @fmaximum)
  call void(i32, i32, ptr, double, i32, i32, double, ptr, ...) @llvm.kit.reduce(i32 512, i32 18, ptr %result, double %d, i32 2, i32 8, double 0.000000E+00, ptr @fmaximumnum)
  call void(i32, i32, ptr, double, i32, i32, double, ptr, ...) @llvm.kit.reduce(i32 512, i32 20, ptr %result, double %d, i32 2, i32 8, double 0.000000E+00, ptr @fmin)
  call void(i32, i32, ptr, double, i32, i32, double, ptr, ...) @llvm.kit.reduce(i32 512, i32 21, ptr %result, double %d, i32 2, i32 8, double 0.000000E+00, ptr @fminimum)
  call void(i32, i32, ptr, double, i32, i32, double, ptr, ...) @llvm.kit.reduce(i32 512, i32 22, ptr %result, double %d, i32 2, i32 8, double 0.000000E+00, ptr @fminimumnum)
  call void(i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce(i32 512, i32 24, ptr %result, i32 %i, i32 2, i32 4, i32 0, ptr @smax)
  call void(i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce(i32 512, i32 25, ptr %result, i32 %i, i32 2, i32 4, i32 0, ptr @smin)
  call void(i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce(i32 512, i32 26, ptr %result, i32 %i, i32 2, i32 4, i32 0, ptr @umax)
  call void(i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce(i32 512, i32 27, ptr %result, i32 %i, i32 2, i32 4, i32 0, ptr @umin)
  ret void
}

; CHECK-LABEL: @g
; CHECK-NEXT: call void {{.+}} @llvm.kit.reduce.i32(i32 2,
; CHECK-NEXT: call void {{.+}} @llvm.kit.reduce.i32(i32 4,
; CHECK-NEXT: ret void
define void @g(ptr %result, i32 %i) {
  call void(i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce(i32 2, i32 5, ptr %result, i32 %i, i32 2, i32 4, i32 0, ptr @add)
  call void(i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce(i32 4, i32 5, ptr %result, i32 %i, i32 2, i32 4, i32 0, ptr @add)
  ret void
}

declare void @and(ptr, i32)
declare void @or(ptr, i32)
declare void @xor(ptr, i32)
declare void @add(ptr, i32)
declare void @fadd(ptr, double)
declare void @fmax(ptr, double)
declare void @fmaximum(ptr, double)
declare void @fmaximumnum(ptr, double)
declare void @fmin(ptr, double)
declare void @fminimum(ptr, double)
declare void @fminimumnum(ptr, double)
declare void @smax(ptr, i32)
declare void @smin(ptr, i32)
declare void @umax(ptr, i32)
declare void @umin(ptr, i32)
