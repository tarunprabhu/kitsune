; Check that the kit-prepare pass works as expected with nested non-reduction
; tapir loops of depth 2.
;
; RUN: opt -passes="kit-prepare" -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SYNCREG_I:.+]] = tail call token @llvm.syncregion.start
; CHECK-NEXT: br label %[[PH_WRAP_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH_WRAP_I]]:
; CHECK-NEXT: %[[NUMTHRDS_I:.+]] = call i64 @llvm.kit.cpu.num.threads(i32 512)
; CHECK-NEXT: br label %[[HEADER_WRAP_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_WRAP_I]]:
; CHECK-NEXT: %[[IV_WRAP_I:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH_WRAP_I]] ]
; CHECK-SAME: [ %[[NEXT_IV_WRAP_I:.+]], %[[LATCH_WRAP_I:.+]] ]
; CHECK-NEXT: detach within %[[SYNCREG_I]], label %[[BODY_WRAP_I:.+]], label %[[LATCH_WRAP_I]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_WRAP_I]]:
; CHECK-NEXT: %[[SYNCREG_J:.+]] = tail call token @llvm.syncregion.start
; CHECK: %[[PER_THRD_I:.+]] = udiv {{.+}} %[[NUMTHRDS_I]]
; CHECK-NEXT: %[[START_I:.+]] = mul i64 %[[IV_WRAP_I]], %[[PER_THRD_I]]
; CHECK-NEXT: %[[ENDMAX_I:.+]] = add i64 %[[START_I]], %[[PER_THRD_I]]
; CHECK-NEXT: %[[STOP_I:.+]] = call i64 @llvm.umin.i64(i64 %[[ENDMAX_I]], i64 %[[N]])
; CHECK-NEXT: %[[GUARD_I:.+]] = icmp uge i64 %[[START_I]], %[[STOP_I]]
; CHECK-NEXT: br i1 %[[GUARD_I]], label %[[END_I:.+]], label %[[PH_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH_I]]:
; CHECK-NEXT: br label %[[HEADER_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_I]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ %[[START_I]], %[[PH_I]] ]
; CHECK-SAME: [ %[[NEXT_I:.+]], %[[LATCH_I:.+]] ]
; CHECK-NEXT: br label %[[PH_J:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH_J]]:
; CHECK-NEXT: br label %[[PH_WRAP_J:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH_WRAP_J]]:
; CHECK-NEXT: %[[NUMTHRDS_J:.+]] = call i64 @llvm.kit.cpu.num.threads(i32 512)
; CHECK-NEXT: br label %[[HEADER_WRAP_J:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_WRAP_J]]:
; CHECK-NEXT: %[[IV_WRAP_J:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH_WRAP_J]] ]
; CHECK-SAME: [ %[[NEXT_IV_WRAP_J:.+]], %[[LATCH_WRAP_J:.+]] ]
; CHECK-NEXT: detach within %[[SYNCREG_J]], label %[[BODY_WRAP_J:.+]], label %[[LATCH_WRAP_J]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_WRAP_J]]:
; CHECK: %[[PER_THRD_J:.+]] = udiv {{.+}} %[[NUMTHRDS_J]]
; CHECK-NEXT: %[[START_J:.+]] = mul i64 %[[IV_WRAP_J]], %[[PER_THRD_J]]
; CHECK-NEXT: %[[ENDMAX_J:.+]] = add i64 %[[START_J]], %[[PER_THRD_J]]
; CHECK-NEXT: %[[STOP_J:.+]] = call i64 @llvm.umin.i64(i64 %[[ENDMAX_J]], i64 %[[N]])
; CHECK-NEXT: %[[GUARD_J:.+]] = icmp uge i64 %[[START_J]], %[[STOP_J]]
; CHECK-NEXT: br i1 %[[GUARD_J]], label %[[END_J:.+]], label %[[PH_J:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH_J]]:
; CHECK-NEXT: br label %[[HEADER_J:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_J]]:
; CHECK-NEXT: %[[J:.+]] = phi i64
; CHECK-SAME: [ %[[START_J]], %[[PH_J]] ]
; CHECK-SAME: [ %[[NEXT_J:.+]], %[[LATCH_J:.+]] ]
; CHECK-NEXT: br label %[[BODY_J:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_J]]:
; CHECK-NEXT: br label %[[LATCH_J]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_J]]:
; CHECK-NEXT: %[[NEXT_J]] = add i64 %[[J]], 1
; CHECK-NEXT: %[[CMP_J:.+]] = icmp eq i64 %[[NEXT_J]], %[[STOP_J]]
; CHECK-NEXT: br i1 %[[CMP_J]], label %[[EXIT_J:.+]], label %[[HEADER_J]]
; CHECK-SAME: !llvm.loop ![[LOOP_J:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT_J]]:
; CHECK-NEXT: br label %[[END_J:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END_J]]:
; CHECK-NEXT: br label %[[REATTACH_WRAP_J:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH_WRAP_J]]:
; CHECK-NEXT: reattach within %[[SYNCREG_J]], label %[[LATCH_WRAP_J]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_WRAP_J]]:
; CHECK-NEXT: %[[NEXT_IV_WRAP_J:.+]] = add i64 %[[IV_WRAP_J:.+]], 1
; CHECK-NEXT: %[[CMP_WRAP_J:.+]] = icmp eq i64 %[[NEXT_IV_WRAP_J]], %[[NUMTHRDS_J]]
; CHECK-NEXT: br i1 %[[CMP_WRAP_J]], label %[[EXIT_WRAP_J:.+]], label %[[HEADER_WRAP_J]]
; CHECK-SAME: !llvm.loop ![[LOOP_WRAP_J:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT_WRAP_J]]:
; CHECK-NEXT: br label %[[END_WRAP_J:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END_WRAP_J]]:
; CHECK-NEXT: sync within %[[SYNCREG_J]], label %[[END_WRAP_J2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END_WRAP_J2]]:
; CHECK-NEXT: br label %[[LATCH_I]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_I]]:
; CHECK-NEXT: %[[NEXT_I:.+]] = add i64 %[[I]], 1
; CHECK-NEXT: %[[CMP_I:.+]] = icmp eq i64 %[[NEXT_I]], %[[STOP_I]]
; CHECK-NEXT: br i1 %[[CMP_I]], label %[[EXIT_I:.+]], label %[[HEADER_I]]
; CHECK-SAME: !llvm.loop ![[LOOP_I:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT_I]]:
; CHECK-NEXT: br label %[[END_I]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END_I]]:
; CHECK-NEXT: br label %[[REATTACH_WRAP_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH_WRAP_I]]:
; CHECK-NEXT: reattach within %[[SYNCREG_I]], label %[[LATCH_WRAP_I]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_WRAP_I]]:
; CHECK-NEXT: %[[NEXT_IV_WRAP_I]] = add i64 %[[IV_WRAP_I]], 1
; CHECK-NEXT: %[[CMP_WRAP_I:.+]] = icmp eq i64 %[[NEXT_IV_WRAP_I]], %[[NUMTHRDS_I]]
; CHECK-NEXT: br i1 %[[CMP_WRAP_I]], label %[[EXIT_WRAP_I:.+]], label %[[HEADER_WRAP_I]]
; CHECK-SAME: !llvm.loop ![[LOOP_WRAP_I:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT_WRAP_I]]
; CHECK-NEXT: br label %[[END_WRAP_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END_WRAP_I]]:
; CHECK-NEXT: sync within %[[SYNCREG_I]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
;
; CHECK-DAG: ![[TARGET:.+]] = !{!"tapir.loop.target", i32 512}
; CHECK-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[LOOP_WRAP_I]] = distinct !{![[LOOP_WRAP_I]], ![[TARGET]], ![[PREPARED]]}
; CHECK-DAG: ![[LOOP_WRAP_J]] = distinct !{![[LOOP_WRAP_J]], ![[TARGET]], ![[PREPARED]]}
; CHECK-DAG: ![[LOOP_I]] = distinct !{![[LOOP_I]]}
; CHECK-DAG: ![[LOOP_J]] = distinct !{![[LOOP_J]]}

define void @f(i64 %n) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.j.ph, label %for.i.latch

for.j.ph:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.j.ph ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 512}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0}
