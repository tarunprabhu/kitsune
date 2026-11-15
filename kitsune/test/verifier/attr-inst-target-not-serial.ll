; The only value allowed for the target attribute on an instruction is 'serial'.
; This is a temporary restriction, and should be removed at some point.
;
; RUN: not opt -o /dev/null %s 2>&1 | FileCheck %s
;
; CHECK: the only target allowed on instructions is 'serial'

define void @f() {
  %sr = call token @llvm.syncregion.start()
  sync within %sr, label %exit, !kit.inst !0

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"kit.inst.target", i32 1024}
