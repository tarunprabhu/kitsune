; Check that the serialize pass reports the loop that was serialized together
; with location information, if it is available.
;
; RUN: opt -passes="kit-serialize" %s -S 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,REMARK
;
; RUN: opt -passes="kit-serialize" %s -S \
; RUN:     -serialize-verbose=1 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,REMARK --allow-empty
;
; REMARK: serialized loop
; REMARK-NEXT: from loop 'loop.j'
; REMARK-NEXT: from function 'pep'
; REMARK-NOT: Loop at depth 2
;
; ------------------------------------------------------------------------------
; RUN: opt -passes="kit-serialize" %s -S \
; RUN:     -serialize-verbose=0 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,QUIET --allow-empty
;
; QUIET-NOT: serialized loop
;
; ------------------------------------------------------------------------------
; RUN: opt -passes="kit-serialize" %s -S \
; RUN:     -serialize-verbose=2 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,VERBOSE --allow-empty
;
; VERBOSE: serialized loop
; VERBOSE-NEXT: from loop 'loop.j'
; VERBOSE-NEXT: from function 'pep'
; VERBOSE-NEXT: Loop at depth 2
;
; ------------------------------------------------------------------------------
;
; ALL: %syncreg.i = tail call token @llvm.syncregion.start()
; ALL: %i = phi i64
; ALL: detach within %syncreg.i
; ALL-NOT: tail call token @llvm.syncregion.start()
; ALL: call void @ext1
; ALL: %j = phi i64
; ALL-NOT: detach within %syncreg.j
; ALL: call void @ext2
; ALL-NOT: reattach within %syncreg.j
; ALL-NOT: sync within %syncreg.j
; ALL: reattach within %syncreg.i
; ALL: sync within %syncreg.i

; forall (i ...) {
;     ext1(i);
;     forall (j ...) {
;         ext2(i, j);
;     }
; }
define void @pep(i64 %m, i64 %n) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  tail call void @ext1(i64 %i)
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  tail call void @ext2(i64 %i, i64 %j)
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %j.not = icmp eq i64 %inc.j, %n
  br i1 %j.not, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %i.not = icmp eq i64 %inc.i, %m
  br i1 %i.not, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

declare void @ext1(i64)

declare void @ext2(i64, i64)

!kit.module.loops.annotated = !{}

!0 = distinct !{!0, !2, !3, !4, !6}
!1 = distinct !{!1, !2, !5}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.perfect.depth", i32 1}
!4 = !{!"tapir.loop.perfect.level", i32 1}
!5 = !{!"loop.name", !"loop.j"}
!6 = !{!"loop.name", !"loop.i"}
