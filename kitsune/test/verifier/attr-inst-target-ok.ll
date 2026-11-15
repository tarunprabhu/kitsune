; The target attribute is only allowed on detach, reattach, and sync
; instructions.
;
; RUN: llvm-as -o /dev/null %s 2>&1 | FileCheck %s --allow-empty
;
; CHECK-NOT: {{^.+$}}

define void @f() {
  %sr = call token @llvm.syncregion.start()
  br label %beg

beg:
  detach within %sr, label %body, label %cont, !kit.inst !0

body:
  br label %end

end:
  reattach within %sr, label %cont, !kit.inst !1

cont:
  sync within %sr, label %exit, !kit.inst !2

exit:
  ret void
}

!0 = distinct !{!0, !3}
!1 = distinct !{!1, !3}
!2 = distinct !{!2, !3}
!3 = !{!"kit.inst.target", i32 1}
