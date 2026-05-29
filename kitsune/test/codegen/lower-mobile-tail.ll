; Check that the tail call attribute is preserved when lowering Kitsune's mobile
; intrinsics. The handling is the same for all tapir targets, so checking this
; with the 'serial' tapir target is sufficient.
;
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s

target triple = "x86_64-pc-linux-gnu"

; CHECK-LABEL: @allocate
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK: tail call noalias ptr @malloc(i64 %[[N]])
; CHECK: notail call noalias ptr @malloc(i64 %[[N]])
; CHECK: tail call noalias ptr @malloc(i64 %[[N]])
define ptr addrspace(67) @allocate(i64 %n) {
  %1 = tail call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %n)
  %2 = notail call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %n)
  %3 = musttail call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i32 1, i64 %n)
  ret ptr addrspace(67) %3
}

; CHECK-LABEL: @deallocate
; CHECK-SAME: ptr addrspace(67) %[[P:[^)]+]]
; CHECK: tail call void @free(
; CHECK: notail call void @free(
; CHECK: tail call void @free(
define void @deallocate(ptr addrspace(67) %p) {
  tail call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %p)
  notail call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %p)
  musttail call void @llvm.kit.mobile.free(i32 1, ptr addrspace(67) %p)
  ret void
}

; Since llvm.kit.mobile.init is a vararg function, LLVM does not allow tail
; attributes, 'tail' and 'musttail'. 'notail' is ok.
; CHECK-LABEL: @init
; CHECK-SAME: ptr addrspace(67) %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: ptr %[[V:[^)]+]]
; CHECK: notail call void @__kitrt_mobile_init_from(
define void @init(ptr addrspace(67) %buf, i64 %n, ptr %v) {
  notail call void (i32, ptr addrspace(67), i64, ptr, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, ptr %v, i32 64)
  ret void
}
