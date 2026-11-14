; Check that the kit-lower-intrinsics-early pass only lowers calls to
; @llvm.kit.cpu.num.threads when the TTID is 'serial'
;
; RUN: opt -passes='kit-lower-intrinsics-early' -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: %[[OPENCILK:.+]] = call i64 @llvm.kit.cpu.num.threads(i32 8)
; CHECK-NEXT: %[[OPENMP:.+]] = call i64 @llvm.kit.cpu.num.threads(i32 512)
; CHECK-NEXT: %[[PTHREADS:.+]] = call i64 @llvm.kit.cpu.num.threads(i32 1024)
; CHECK-NEXT: %[[QTHREADS:.+]] = call i64 @llvm.kit.cpu.num.threads(i32 32)
; CHECK-NEXT: call void @ext(i64 1)
; CHECK-NEXT: call void @ext(i64 %[[OPENCILK]])
; CHECK-NEXT: call void @ext(i64 %[[OPENMP]])
; CHECK-NEXT: call void @ext(i64 %[[PTHREADS]])
; CHECK-NEXT: call void @ext(i64 %[[QTHREADS]])

define void @f() {
  %serial = call i64 @llvm.kit.cpu.num.threads(i32 1)
  %opencilk = call i64 @llvm.kit.cpu.num.threads(i32 8)
  %openmp = call i64 @llvm.kit.cpu.num.threads(i32 512)
  %pthreads = call i64 @llvm.kit.cpu.num.threads(i32 1024)
  %qthreads = call i64 @llvm.kit.cpu.num.threads(i32 32)
  call void @ext(i64 %serial)
  call void @ext(i64 %opencilk)
  call void @ext(i64 %openmp)
  call void @ext(i64 %pthreads)
  call void @ext(i64 %qthreads)
  ret void
}

declare void @ext(i64)
