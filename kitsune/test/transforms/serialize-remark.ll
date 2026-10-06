; Check that the -serialize-verbose option works as expected.
;
; ------------------------------------------------------------------------------
; For now, the default is to emit remarks, but this might change in the future.
;
; RUN: opt -passes=kit-serialize -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=V1
;
; ------------------------------------------------------------------------------
; RUN: opt -passes=kit-serialize -serialize-verbose=0 -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --allow-empty --check-prefix=V0
;
; V0-NOT: {{^.+$}}
;
; ------------------------------------------------------------------------------
; RUN: opt -passes=kit-serialize -serialize-verbose -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=V1
;
; V1: serialized loop
; V1-NEXT: from loop 'header4'
; V1-NEXT: from function 'f'

define void @f(i64 %n) !kit.func !0 {
entry:
  %sr1 = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %sr2 = call token @llvm.syncregion.start()
  br label %header2

header2:
  %j = phi i64 [ 0, %body1 ], [ %inc2, %latch2 ]
  detach within %sr2, label %body2, label %latch2

body2:
  %sr3 = call token @llvm.syncregion.start()
  br label %header3

header3:
  %k = phi i64 [ 0, %body2 ], [ %inc3, %latch3 ]
  detach within %sr3, label %body3, label %latch3

body3:
  %sr4 = call token @llvm.syncregion.start()
  br label %header4

header4:
  %l4 = phi i64 [ 0, %body3 ], [ %inc4, %latch4 ]
  detach within %sr4, label %body4, label %latch4

body4:
  reattach within %sr4, label %latch4

latch4:
  %inc4 = add i64 %l4, 1
  %cmp4 = icmp eq i64 %inc4, %n
  br i1 %cmp4, label %sync4, label %header4, !llvm.loop !5

sync4:
  sync within %sr4, label %reattach3

reattach3:
  reattach within %sr3, label %latch3

latch3:
  %inc3 = add i64 %k, 1
  %cmp3 = icmp eq i64 %inc3, %n
  br i1 %cmp3, label %sync3, label %header3, !llvm.loop !4

sync3:
  sync within %sr3, label %reattach2

reattach2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %j, 1
  %cmp2 = icmp eq i64 %inc2, %n
  br i1 %cmp2, label %sync2, label %header2, !llvm.loop !3

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !2

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"kit.func.pre.lower.annotate.pass"}
!2 = distinct !{!2, !6, !7, !8}
!3 = distinct !{!3, !6, !9}
!4 = distinct !{!4, !6, !10}
!5 = distinct !{!5, !6, !11}
!6 = !{!"tapir.loop.target", i32 1}
!7 = !{!"tapir.loop.perfect.depth", i32 4}
!8 = !{!"tapir.loop.perfect.level", i32 1}
!9 = !{!"tapir.loop.perfect.level", i32 2}
!10 = !{!"tapir.loop.perfect.level", i32 3}
!11 = !{!"tapir.loop.perfect.level", i32 4}
