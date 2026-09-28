; Check that a tapir loop of depth 2 is prepared as expected. In this case, the
; top-level loop remains unchanged, but the inner loop is transformed.
;
; RUN: opt --tapir=cuda --passes=kit-prepare -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[M:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
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
; CHECK-NEXT: br label %[[PH_WRAP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH_WRAP]]:
; CHECK-NEXT: %[[IPT:.+]] = call i64 @llvm.kit.gpu.ipt(i32 2, ptr null)
; CHECK-NEXT: %[[IPT_MINUS_1:.+]] = sub i64 %[[IPT]], 1
; CHECK-NEXT: %[[N_PLUS_IPT_MINUS_1:.+]] = add i64 %[[N]], %[[IPT_MINUS_1]]
; CHECK-NEXT: %[[OTC:.+]] = udiv i64 %[[N_PLUS_IPT_MINUS_1]], %[[IPT]]
; CHECK-NEXT: br label %[[HEADER_WRAP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_WRAP]]:
; CHECK-NEXT: %[[IV_WRAP:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH_WRAP]] ]
; CHECK-SAME: [ %[[NEXTV_WRAP:.+]], %[[LATCH_WRAP:.+]] ]
; CHECK-NEXT: detach within %[[SRJ]], label %[[BODY_WRAP:.+]], label %[[LATCH_WRAP]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_WRAP]]:
; CHECK-NEXT: %[[THRDS_PLUS_N:.+]] = add i64 %[[N]], %[[OTC]]
; CHECK-NEXT: %[[THRDS_PLUS_N_1:.+]] = sub i64 %[[THRDS_PLUS_N]], 1
; CHECK-NEXT: %[[PER_THRD:.+]] = udiv i64 %[[THRDS_PLUS_N_1]], %[[OTC]]
; CHECK-NEXT: %[[START:.+]] = mul i64 %[[IV_WRAP]], %[[PER_THRD]]
; CHECK-NEXT: %[[ENDMAX:.+]] = add i64 %[[START]], %[[PER_THRD]]
; CHECK-NEXT: %[[STOP:.+]] = call i64 @llvm.umin.i64(i64 %[[ENDMAX]], i64 %[[N]])
; CHECK-NEXT: %[[GUARD:.+]] = icmp uge i64 %[[START]], %[[STOP]]
; CHECK-NEXT: br i1 %[[GUARD]], label %[[END:.+]], label %[[PHJ:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PHJ]]:
; CHECK-NEXT: br label %[[HEADERJ:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADERJ]]:
; CHECK-NEXT: %[[J:.+]] = phi i64
; CHECK-SAME: [ %[[START]], %[[PHJ]] ]
; CHECK-SAME: [ %[[NEXT:.+]], %[[LATCHJ:.+]] ]
; CHECK-NEXT: br label %[[BODYJ:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODYJ]]:
; CHECK-NEXT: br label %[[LATCHJ]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCHJ]]:
; CHECK-NEXT: %[[NEXTJ:.+]] = add i64 %[[J]], 1
; CHECK-NEXT: %[[CMPJ:.+]] = icmp eq i64 %[[NEXTJ]], %[[STOP]]
; CHECK-NEXT: br i1 %[[CMPJ]], label %[[EXITJ:.+]], label %[[HEADERJ]]
; CHECK-SAME: !llvm.loop ![[LOOPJ:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXITJ]]:
; CHECK-NEXT: br label %[[ENDJ:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ENDJ]]:
; CHECK-NEXT: br label %[[REATTACH_WRAP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH_WRAP]]:
; CHECK-NEXT: reattach within %[[SRJ]], label %[[LATCH_WRAP]]
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
; CHECK-NEXT: sync within %[[SRJ]], label %[[REATTACHJ:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACHJ]]:
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
; CHECK-DAG: ![[LOOPJ]] = distinct !{![[LOOPJ]]}
; CHECK-DAG: ![[LOOPI]] = distinct !{![[LOOPI]], {{.+}}, ![[PREPARED]]}

define void @f(i64 %m, i64 %n) {
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

!0 = distinct !{!0, !2, !3, !4}
!1 = distinct !{!1, !2, !5}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.perfect.depth", i32 2}
!4 = !{!"tapir.loop.perfect.level", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 2}
