; If the tool is run on already transformed code, it must not change the
; output. This is effectively the same as testing that sorted code remains
; unaffected.
;
; RUN: cat %s | %kit-sort | %kit-sort | FileCheck %s

; CHECK-LABEL: @f
; CHECK: entry:
; CHECK: check:
; CHECK: if.false:
; CHECK: if.true:
; CHECK: exit:
define void @f(i1 %cond) {
entry:
  br label %check

exit:
  ret void

check:
  br i1 %cond, label %if.true, label %if.false

if.false:
  br label %exit

if.true:
  br label %exit
}
