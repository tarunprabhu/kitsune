; Check that a tapir loop containing an exit to a dead-end block is lowered
; correctly. While we don't allow early exits in parallel loops in the frontend,
; branches to dead-end blocks can occur as a result of certain optimization
; passes.
;
; RUN: opt -passes=kit-prepare -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: i64 %[[B:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[PH_WRAP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH_WRAP]]:
; CHECK-NEXT: %[[NUMTHRDS:.+]] = call i64 @llvm.kit.cpu.num.threads(i32 512)
; CHECK-NEXT: br label %[[HEADER_WRAP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_WRAP]]:
; CHECK-NEXT: %[[IV_WRAP:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH_WRAP]] ]
; CHECK-SAME: [ %[[NEXTV_WRAP:.+]], %[[LATCH_WRAP:.+]] ]
; CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY_WRAP:.+]], label %[[LATCH_WRAP]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_WRAP]]:
; CHECK-NEXT: %[[THRDS_PLUS_N:.+]] = add i64 %[[N]], %[[NUMTHRDS]]
; CHECK-NEXT: %[[THRDS_PLUS_N_1:.+]] = sub i64 %[[THRDS_PLUS_N]], 1
; CHECK-NEXT: %[[PER_THRD:.+]] = udiv i64 %[[THRDS_PLUS_N_1]], %[[NUMTHRDS]]
; CHECK-NEXT: %[[START:.+]] = mul i64 %[[IV_WRAP]], %[[PER_THRD]]
; CHECK-NEXT: %[[ENDMAX:.+]] = add i64 %[[START]], %[[PER_THRD]]
; CHECK-NEXT: %[[STOP:.+]] = call i64 @llvm.umin.i64(i64 %[[ENDMAX]], i64 %[[N]])
; CHECK-NEXT: %[[GUARD:.+]] = icmp uge i64 %[[START]], %[[STOP]]
; CHECK-NEXT: br i1 %[[GUARD]], label %[[END:.+]], label %[[PH:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH]]:
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ %[[START]], %[[PH]] ]
; CHECK-SAME: [ %[[NEXT:.+]], %[[LATCH:.+]] ]
; CHECK-NEXT: br label %[[BODY:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: %[[CHK:.+]] = icmp eq i64 %[[I]], %[[B]]
; CHECK-NEXT: br i1 %[[CHK]], label %[[DEADEND:.+]], label %[[REATTACH:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH]]:
; CHECK-NEXT: br label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[NEXT:.+]] = add i64 %[[I]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[NEXT]], %[[STOP]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-SAME: !llvm.loop ![[LOOP:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: br label %[[END]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END]]:
; CHECK-NEXT: br label %[[REATTACH_WRAP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH_WRAP]]:
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH_WRAP]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_WRAP]]:
; CHECK-NEXT: %[[NEXTV_WRAP]] = add i64 %[[IV_WRAP]], 1
; CHECK-NEXT: %[[CMP_WRAP:.+]] = icmp eq i64 %[[NEXTV_WRAP]], %[[NUMTHRDS]]
; CHECK-NEXT: br i1 %[[CMP_WRAP]], label %[[EXIT_WRAP:.+]], label %[[HEADER_WRAP]]
; CHECK-SAME: !llvm.loop ![[LOOP_WRAP:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT_WRAP]]
; CHECK-NEXT: br label %[[END_WRAP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END_WRAP]]:
; CHECK-NEXT: sync within %[[SYNCREG]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[DEADEND]]:
; CHECK-NEXT: unreachable
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
;
; CHECK-DAG: ![[TARGET:.+]] = !{!"tapir.loop.target", i32 512}
; CHECK-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[LOOP_WRAP]] = distinct !{![[LOOP_WRAP]], ![[TARGET]], ![[PREPARED]]}
; CHECK-DAG: ![[LOOP]] = distinct !{![[LOOP]]}

define void @f(i64 %n, i64 %b) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  %chk = icmp eq i64 %i, %b
  br i1 %chk, label %deadend, label %for.i.reattach

for.i.reattach:
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  sync within %syncreg, label %for.i.end

deadend:
  unreachable

for.i.end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 512}
!1 = distinct !{!1, !0}
