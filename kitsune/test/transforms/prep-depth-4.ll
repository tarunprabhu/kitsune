; Only Tapir loop nests with depth <= 3 can be prepared.
;
; RUN: not opt -passes='kit-prepare' -S %s 2>&1 | FileCheck %s
;
; CHECK: depth of tapir loop nest must be in range [1,3]

define void @f(i64 %n1, i64 %n2, i64 %n3, i64 %n4) {
entry:
  %sr1 = call token @llvm.syncregion.start()
  %cmp.n1 = icmp ult i64 0, %n1
  br i1 %cmp.n1, label %ph1, label %sync1

ph1:
  br label %header1

header1:
  %i = phi i64 [ 0, %ph1 ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %sr2 = call token @llvm.syncregion.start()
  %cmp.n2 = icmp ult i64 0, %n2
  br i1 %cmp.n2, label %ph2, label %sync2

ph2:
  br label %header2

header2:
  %j = phi i64 [ 0, %ph2 ], [ %inc2, %latch2 ]
  detach within %sr2, label %body2, label %latch2

body2:
  %sr3 = call token @llvm.syncregion.start()
  %cmp.n3 = icmp ult i64 0, %n3
  br i1 %cmp.n3, label %ph3, label %sync3

ph3:
  br label %header3

header3:
  %k = phi i64 [ 0, %ph3 ], [ %inc3, %latch3 ]
  detach within %sr3, label %body3, label %latch3

body3:
  %sr4 = call token @llvm.syncregion.start()
  %cmp.n4 = icmp ult i64 0, %n4
  br i1 %cmp.n4, label %ph4, label %sync4

ph4:
  br label %header4

header4:
  %l4 = phi i64 [ 0, %ph4 ], [ %inc4, %latch4 ]
  detach within %sr4, label %body4, label %latch4

body4:
  reattach within %sr4, label %latch4

latch4:
  %inc4 = add i64 %l4, 1
  %cmp4 = icmp eq i64 %inc4, %n4
  br i1 %cmp4, label %loopexit4, label %header4, !llvm.loop !6

loopexit4:
  br label %sync4

sync4:
  sync within %sr4, label %reattach3

reattach3:
  reattach within %sr3, label %latch3

latch3:
  %inc3 = add i64 %k, 1
  %cmp3 = icmp eq i64 %inc3, %n3
  br i1 %cmp3, label %loopexit3, label %header3, !llvm.loop !7

loopexit3:
  br label %sync3

sync3:
  sync within %sr3, label %reattach2

reattach2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %j, 1
  %cmp2 = icmp eq i64 %inc2, %n2
  br i1 %cmp2, label %loopexit2, label %header2, !llvm.loop !8

loopexit2:
  br label %sync2

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i, 1
  %cmp1 = icmp eq i64 %inc1, %n1
  br i1 %cmp1, label %loopexit1, label %header1, !llvm.loop !9

loopexit1:
  br label %sync1

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!"tapir.loop.perfect.depth", i32 4}
!2 = !{!"tapir.loop.perfect.level", i32 1}
!3 = !{!"tapir.loop.perfect.level", i32 2}
!4 = !{!"tapir.loop.perfect.level", i32 3}
!5 = !{!"tapir.loop.perfect.level", i32 4}
!6 = distinct !{!6, !0, !5}
!7 = distinct !{!7, !0, !4}
!8 = distinct !{!8, !0, !3}
!9 = distinct !{!9, !0, !1, !2}
