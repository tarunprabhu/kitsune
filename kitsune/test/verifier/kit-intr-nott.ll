; The first argument in calls to Kitsune's intrinsics must be a valid TTID.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: first argument to call is not a valid TTID

define void @f() {
  call void @llvm.kit.mobile.free(i32 -1, ptr addrspace(67) null)
  ret void
}
