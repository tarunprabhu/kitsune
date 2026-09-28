; Non-PHI instructions are not allowed in the tapir loop header.
;
; FIXME: This should be an error, but it has been changed to a warning because
; it causes too many codes to break. The underlying issue can probably be fixed
; by making some LLVM passes - notably the loop vectorizer - tapir-aware. Until
; then, we issue a warning instead of an error; the way Kitsune currently works,
; the presence of any non-PHI instructions in the header does not result in
; incorrect code generation. But, obviously, this will not always be the case.
;
; RUN: opt -passes=kit-verify-prelower %s -disable-output 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: warning: tapir loop header contains unexpected instruction '{{.+}}'

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  %i.trunc = trunc i64 %i to i32
  detach within %syncreg, label %body, label %latch

body:
  call void @ext(i32 %i.trunc)
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  sync within %syncreg, label %end

end:
  ret void
}

declare void @ext(i32)

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
