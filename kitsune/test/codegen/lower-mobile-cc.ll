; Check that the calling convention is preserved when lowering Kitsune's
; mobile intrinsics. The handling is the same for all tapir targets, so
; checking this with the 'serial' tapir target is sufficient.
;
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s

target triple = "x86_64-pc-linux-gnu"

; CHECK-LABEL: @allocate
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK: call fastcc noalias ptr @__kitrt_default_mem_alloc(i64 %[[N]])
; CHECK: call coldcc noalias ptr @__kitrt_default_mem_alloc(i64 %[[N]])
; CHECK: call anyregcc noalias ptr @__kitrt_default_mem_alloc(i64 %[[N]])
define void @allocate(i64 %n) {
  %1 = call fastcc noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %n)
  %2 = call coldcc noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %n)
  %3 = call anyregcc noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %n)
  ret void
}

; CHECK-LABEL: @deallocate
; CHECK-SAME: ptr addrspace(67) %[[P:[^)]+]]
; CHECK: call fastcc void @__kitrt_default_mem_free(
; CHECK: call coldcc void @__kitrt_default_mem_free(
; CHECK: call anyregcc void @__kitrt_default_mem_free(
define void @deallocate(ptr addrspace(67) %p) {
  call fastcc void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %p)
  call coldcc void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %p)
  call anyregcc void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %p)
  ret void
}

; CHECK-LABEL: @init
; CHECK-SAME: ptr addrspace(67) %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: double %[[V:[^)]+]]
; CHECK: call fastcc void @__kitrt_mobile_init_double(
; CHECK: call coldcc void @__kitrt_mobile_init_double(
; CHECK: call anyregcc void @__kitrt_mobile_init_double(
define void @init(ptr addrspace(67) %buf, i64 %n, double %v) {
  call fastcc void (i32, ptr addrspace(67), i64, double, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, double %v)
  call coldcc void (i32, ptr addrspace(67), i64, double, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, double %v)
  call anyregcc void (i32, ptr addrspace(67), i64, double, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, double %v)
  ret void
}
