; Check that a simple reduction loop of depth 1 is prepared as expected.
;
; RUN: opt --tapir=opencilk --passes=kit-prepare -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK: %[[RESULT:.+]] = alloca float
; CHECK: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[PH_O:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH_O]]:
; CHECK-NEXT: %[[NREDS:.+]] = call i64 @llvm.kit.cpu.num.threads(i32 8)
; CHECK-NEXT: br label %[[HEADER_O:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_O]]:
; CHECK-NEXT: %[[IV_O:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH_O]] ],
; CHECK-SAME: [ %[[INC_O:.+]], %[[LATCH_O:.+]] ]
; CHECK-NEXT: detach within %[[SYNCREG]], label %[[GUARD_I:.+]], label %[[LATCH_O:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[GUARD_I]]:
; CHECK-NEXT: %[[NPP:.+]] = add {{.+}} %[[N]], %[[NREDS]]
; CHECK-NEXT: %[[NPP_1:.+]] = sub {{.+}} %[[NPP]], 1
; CHECK-NEXT: %[[SZREDS:.+]] = udiv {{.+}} %[[NPP_1]], %[[NREDS]]
; CHECK-NEXT: %[[START:.+]] = mul {{.+}} %[[IV_O]], %[[SZREDS]]
; CHECK-NEXT: %[[PLUS:.+]] = add {{.+}} %[[START]], %[[SZREDS]]
; CHECK-NEXT: %[[END:.+]] = call i64 @llvm.umin.i64(i64 %[[PLUS]], i64 %[[N]])
; CHECK-NEXT: %[[CMP_GUARD:.+]] = icmp uge {{.+}} %[[START]], %[[END]]
; CHECK-NEXT: br i1 %[[CMP_GUARD]], label %[[END_I:.+]], label %[[PH_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH_I]]:
; CHECK-NEXT: %[[LOCAL:.+]] = alloca float
; CHECK-NEXT: store float 0.000000e+00, ptr %[[LOCAL]]
; CHECK-NEXT: br label %[[HEADER_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_I]]:
; CHECK-NEXT: %[[IV_I:.+]] = phi i64
; CHECK-SAME: [ %[[START]], %[[PH_I]] ]
; CHECK-SAME: [ %[[INC_I:.+]], %[[LATCH_I:.+]] ]
; CHECK-NEXT: br label %[[BODY_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_I]]:
; CHECK-NEXT: %[[TRUNC:.+]] = sitofp i64 %[[IV_I]] to float
; CHECK-NEXT: call void @sum(ptr %[[LOCAL]], float %[[TRUNC]])
; CHECK-NEXT: br label %[[LATCH_I]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_I]]:
; CHECK-NEXT: %[[INC_I:.+]] = add i64 %[[IV_I]], 1
; CHECK-NEXT: %[[CMP_I:.+]] = icmp eq i64 %[[INC_I]], %[[END]]
; CHECK-NEXT: br i1 %[[CMP_I]], label %[[EXIT_I:.+]], label %[[HEADER_I]],
; CHECK-SAME: !llvm.loop ![[LOOP_I:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT_I]]:
; CHECK-NEXT: %[[PARTIAL:.+]] = load float, ptr %[[LOCAL]]
; CHECK-NEXT: atomicrmw fadd ptr %[[RESULT]], float %[[PARTIAL]] monotonic
; CHECK-NEXT: br label %[[END_I]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END_I]]:
; CHECK-NEXT: br label %[[REATTACH:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH]]:
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH_O]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_O]]:
; CHECK-NEXT: %[[INC_O:.+]] = add i64 %[[IV_O]], 1
; CHECK-NEXT: %[[CMP_O:.+]] = icmp eq i64 %[[INC_O]], %[[NREDS]]
; CHECK-NEXT: br i1 %[[CMP_O]], label %[[EXIT_O:.+]], label %[[HEADER_O]],
; CHECK-SAME: !llvm.loop ![[LOOP_O:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT_O]]:
; CHECK-NEXT: br label %[[SYNC:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC]]:
; CHECK-NEXT: sync within %[[SYNCREG]],
;
; CHECK-DAG: ![[TARGET:.+]] = !{!"tapir.loop.target", i32 8}
; CHECK-DAG: ![[REDUCTION:.+]] = !{!"tapir.loop.reduction"}
; CHECK-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[LOOP_I]] = distinct !{![[LOOP_I]]}
; CHECK-DAG: ![[LOOP_O]] = distinct !{![[LOOP_O]], ![[REDUCTION]], ![[TARGET]], ![[PREPARED]]}

declare void @sum(ptr %res, float %v)

define void @f(i64 %n) {
entry:
  %result = alloca float
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  %f32 = sitofp i64 %i to float
  call void(i32, i32, ptr, i32, float, float, ptr, ...) @llvm.kit.reduce.0(i32 8, i32 6, ptr %result, i32 4, float %f32, float 0.0, ptr @sum)
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !2

for.i.exit:
  sync within %syncreg, label %for.i.end

for.i.end:
  ret void
}

!0 = !{!"tapir.loop.reduction"}
!1 = !{!"tapir.loop.target", i32 8}
!2 = distinct !{!2, !0, !1}
