; The target attribute is only allowed on detach, reattach, and sync
; instructions. Obviously, it would be ideal if we checked all instructions on
; which the attribute is not allowed, but that would be a bit unwieldy.
;
; RUN: not llvm-as -o /dev/null %s 2>&1 | FileCheck %s
;
; CHECK: attribute 'kit.inst.target' not allowed on 'CondBrInst' instruction

define void @f() {
  br label %header

header:
  %done = call i1 @isdone()
  br i1 %done, label %exit, label %header, !kit.inst !0

exit:
  ret void
}

declare i1 @isdone()

!0 = distinct !{!0, !1}
!1 = !{!"kit.inst.target", i32 1}
