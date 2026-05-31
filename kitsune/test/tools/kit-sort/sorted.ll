; If the blocks in a function are already sorted, this should not change it.
;
; RUN: %kit-sort %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK: entry:
; CHECK: check:
; CHECK: if.false:
; CHECK: if.true:
; CHECK: exit:
define void @f(i1 %cond) {
entry:
  br label %check

check:
  br i1 %cond, label %if.true, label %if.false

if.false:
  br label %exit

if.true:
  br label %exit

exit:
  ret void
}
