; REQUIRES: kitsune-examples
;
; Check that a tapir target plugin works as expected when run with opt. We use
; the tapir target plugin demo for consistency with the way LLVM pass plugins
; are tested.
;
; RUN: opt --tapir=custom --tapir-plugin=%kit-tt-plugin-demo %s \
; RUN:     -S -o - -O2 \
; RUN:     | FileCheck %s --check-prefix=BOOKEND
;
; BOOKEND: call void @bookend
; BOOKEND-NEXT: call {{.*}}void @mset{{[^(]+}}(
; BOOKEND-NEXT: call void @bookend

define void @mset(ptr %a, i64 %n, i64 %v) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4 = icmp sgt i64 %n, 0
  br i1 %cmp4, label %header, label %sync

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr i64, ptr %a, i64 %i
  store i64 %v, ptr %arrayidx, align 8
  reattach within %syncreg, label %latch

latch:
  %i.next = add i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, %n
  br i1 %cmp.i, label %sync, label %header, !llvm.loop !0

sync:
  sync within %syncreg, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 4}
!2 = !{!"tapir.loop.target", i32 2048}
!3 = !{!"tapir.loop.lowering.enabled"}
