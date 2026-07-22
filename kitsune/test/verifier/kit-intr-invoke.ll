; Kitsune's intrinsics cannot be invoked. This will be checked by LLVM's
; standard verifier, but the test is here just to reinforce this. In Kitsune's
; source, we should never have to look for Kitsune's intrinsics in invoke
; instructions.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: Cannot invoke an intrinsic other than

define void @f(ptr addrspace(67) %buf, i64 %n, i32 %init) {
  invoke void (i32, ptr addrspace(67), i64, i32, ...) @llvm.kit.mobile.init(i32 1,  ptr addrspace(67) %buf, i64 %n, i32 %init)
  to label %cont unwind label %lpad

cont:
  br label %exit

lpad:
  %lp = landingpad { ptr, i32 }
  cleanup
  br label %exit

exit:
  ret void
}
