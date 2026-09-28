; Check that a tapir loop nes of depth 3 is prepared as expected. Only the
; innermost loop is transformed, but all loops in the nest are marked as
; prepared.
;
; RUN: opt --tapir=cuda --passes=kit-prepare -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[M:[^,]+]]
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: i64 %[[P:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SRI:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[HEADERI:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADERI]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INCI:.+]], %[[LATCHI:.+]] ]
; CHECK-NEXT: detach within %[[SRI]], label %[[BODYI:.+]], label %[[LATCHI]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODYI]]:
; CHECK-NEXT: %[[SRJ:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[HEADERJ:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADERJ]]:
; CHECK-NEXT: %[[J:.+]] = phi i64
; CHECK-SAME: [ 0, %[[BODYI]] ]
; CHECK-SAME: [ %[[INCJ:.+]], %[[LATCHJ:.+]] ]
; CHECK-NEXT: detach within %[[SRJ]], label %[[BODYJ:.+]], label %[[LATCHJ]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODYJ]]:
; CHECK-NEXT: %[[SRK:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[PH_WRAP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH_WRAP]]:
; CHECK-NEXT: %[[IPT:.+]] = call i64 @llvm.kit.gpu.ipt(i32 2, ptr null)
; CHECK-NEXT: %[[IPT_MINUS_1:.+]] = sub i64 %[[IPT]], 1
; CHECK-NEXT: %[[N_PLUS_IPT_MINUS_1:.+]] = add i64 %[[P]], %[[IPT_MINUS_1]]
; CHECK-NEXT: %[[OTC:.+]] = udiv i64 %[[N_PLUS_IPT_MINUS_1]], %[[IPT]]
; CHECK-NEXT: br label %[[HEADER_WRAP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_WRAP]]:
; CHECK-NEXT: %[[IV_WRAP:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH_WRAP]] ]
; CHECK-SAME: [ %[[NEXTV_WRAP:.+]], %[[LATCH_WRAP:.+]] ]
; CHECK-NEXT: detach within %[[SRK]], label %[[BODY_WRAP:.+]], label %[[LATCH_WRAP]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_WRAP]]:
; CHECK-NEXT: %[[THRDS_PLUS_P:.+]] = add i64 %[[P]], %[[OTC]]
; CHECK-NEXT: %[[THRDS_PLUS_P_1:.+]] = sub i64 %[[THRDS_PLUS_P]], 1
; CHECK-NEXT: %[[PER_THRD:.+]] = udiv i64 %[[THRDS_PLUS_P_1]], %[[OTC]]
; CHECK-NEXT: %[[START:.+]] = mul i64 %[[IV_WRAP]], %[[PER_THRD]]
; CHECK-NEXT: %[[ENDMAX:.+]] = add i64 %[[START]], %[[PER_THRD]]
; CHECK-NEXT: %[[STOP:.+]] = call i64 @llvm.umin.i64(i64 %[[ENDMAX]], i64 %[[P]])
; CHECK-NEXT: %[[GUARD:.+]] = icmp uge i64 %[[START]], %[[STOP]]
; CHECK-NEXT: br i1 %[[GUARD]], label %[[END:.+]], label %[[PHK:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PHK]]:
; CHECK-NEXT: br label %[[HEADERK:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADERK]]:
; CHECK-NEXT: %[[K:.+]] = phi i64
; CHECK-SAME: [ %[[START]], %[[PHK]] ]
; CHECK-SAME: [ %[[NEXT:.+]], %[[LATCHK:.+]] ]
; CHECK-NEXT: br label %[[BODYK:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODYK]]:
; CHECK-NEXT: br label %[[LATCHK]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCHK]]:
; CHECK-NEXT: %[[NEXTK:.+]] = add i64 %[[K]], 1
; CHECK-NEXT: %[[CMPK:.+]] = icmp eq i64 %[[NEXTK]], %[[STOP]]
; CHECK-NEXT: br i1 %[[CMPK]], label %[[EXITK:.+]], label %[[HEADERK]]
; CHECK-SAME: !llvm.loop ![[LOOPK:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXITK]]:
; CHECK-NEXT: br label %[[ENDK:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDK]]:
; CHECK-NEXT: br label %[[REATTACHK:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACHK]]:
; CHECK-NEXT: reattach within %[[SRK]], label %[[LATCH_WRAP]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_WRAP]]:
; CHECK-NEXT: %[[NEXTV_WRAP]] = add i64 %[[IV_WRAP]], 1
; CHECK-NEXT: %[[CMP_WRAP:.+]] = icmp eq i64 %[[NEXTV_WRAP]], %[[OTC]]
; CHECK-NEXT: br i1 %[[CMP_WRAP]], label %[[EXIT_WRAP:.+]], label %[[HEADER_WRAP]]
; CHECK-SAME: !llvm.loop ![[LOOP_WRAP:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT_WRAP]]
; CHECK-NEXT: br label %[[END_WRAP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END_WRAP]]:
; CHECK-NEXT: sync within %[[SRK]], label %[[REATTACHJ:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACHJ]]:
; CHECK-NEXT: reattach within %[[SRJ]], label %[[LATCHJ]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCHJ]]:
; CHECK-NEXT: %[[INCJ]] = add i64 %[[J]], 1
; CHECK-NEXT: %[[CMPJ:.+]] = icmp eq i64 %[[INCJ]], %[[N]]
; CHECK-NEXT: br i1 %[[CMPJ]], label %[[EXITJ:.+]], label %[[HEADERJ]]
; CHECK-SAME: !llvm.loop ![[LOOPJ:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXITJ]]:
; CHECK-NEXT: sync within %[[SRJ]], label %[[ENDJ:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDJ]]:
; CHECK-NEXT: reattach within %[[SRI]], label %[[LATCHI]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCHI]]:
; CHECK-NEXT: %[[INCI]] = add i64 %[[I]], 1
; CHECK-NEXT: %[[CMPI:.+]] = icmp eq i64 %[[INCI]], %[[M]]
; CHECK-NEXT: br i1 %[[CMPI]], label %[[EXITI:.+]], label %[[HEADERI]]
; CHECK-SAME: !llvm.loop ![[LOOPI:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXITI]]:
; CHECK-NEXT: sync within %[[SRI]], label %[[ENDI:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDI]]:
; CHECK-NEXT: ret void
;
; CHECK-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[LOOP_WRAP]] = distinct !{![[LOOP_WRAP]], {{.+}}, ![[PREPARED]]}
; CHECK-DAG: ![[LOOPK]] = distinct !{![[LOOPK]]}
; CHECK-DAG: ![[LOOPJ]] = distinct !{![[LOOPJ]], {{.+}}, ![[PREPARED]]}
; CHECK-DAG: ![[LOOPI]] = distinct !{![[LOOPI]], {{.+}}, ![[PREPARED]]}

define void @f(i64 %m, i64 %n, i64 %p) {
entry:
  %sr.i = call token @llvm.syncregion.start()
  br label %header.i

header.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch.i ]
  detach within %sr.i, label %body.i, label %latch.i

body.i:
  %sr.j = call token @llvm.syncregion.start()
  br label %header.j

header.j:
  %j = phi i64 [ 0, %body.i ], [ %inc.j, %latch.j ]
  detach within %sr.j, label %body.j, label %latch.j

body.j:
  %sr.k = call token @llvm.syncregion.start()
  br label %header.k

header.k:
  %k = phi i64 [ 0, %body.j ], [ %inc.k, %latch.k ]
  detach within %sr.k, label %body.k, label %latch.k

body.k:
  reattach within %sr.k, label %latch.k

latch.k:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %exit.k, label %header.k, !llvm.loop !2

exit.k:
  sync within %sr.k, label %end.k

end.k:
  reattach within %sr.j, label %latch.j

latch.j:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %exit.j, label %header.j, !llvm.loop !1

exit.j:
  sync within %sr.j, label %end.j

end.j:
  reattach within %sr.i, label %latch.i

latch.i:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %exit.i, label %header.i, !llvm.loop !0

exit.i:
  sync within %sr.i, label %end.i

end.i:
  ret void
}

!0 = distinct !{!0, !3, !4, !5}
!1 = distinct !{!1, !3, !6}
!2 = distinct !{!2, !3, !7}
!3 = !{!"tapir.loop.target", i32 2}
!4 = !{!"tapir.loop.perfect.depth", i32 3}
!5 = !{!"tapir.loop.perfect.level", i32 1}
!6 = !{!"tapir.loop.perfect.level", i32 2}
!7 = !{!"tapir.loop.perfect.level", i32 3}
