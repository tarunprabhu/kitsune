; Check that a simple reduction loop of depth 1 is prepared as expected. In
; this case, nothing is changed in the loop body. Only the tapir.loop.prepared
; attribute is added to the tapir reduction loop.
;
; RUN: opt --tapir=serial -passes='kit-prepare' -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK: %[[RESULT:.+]] = alloca i64
; CHECK: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[PH_O:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH_O]]:
; CHECK-NEXT: %[[NREDS:.+]] = call i64 @llvm.kit.cpu.num.threads(i32 1)
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
; CHECK-NEXT: %[[LOCAL:.+]] = alloca i64
; CHECK-NEXT: store i64 0, ptr %[[LOCAL]]
; CHECK-NEXT: br label %[[HEADER_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_I]]:
; CHECK-NEXT: %[[IV_I:.+]] = phi i64
; CHECK-SAME: [ %[[START]], %[[PH_I]] ]
; CHECK-SAME: [ %[[INC_I:.+]], %[[LATCH_I:.+]] ]
; CHECK-NEXT: br label %[[BODY_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_I]]:
; CHECK-NEXT: call void @sum(ptr %[[LOCAL]], i64 %[[IV_I]])
; CHECK-NEXT: br label %[[LATCH_I]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_I]]:
; CHECK-NEXT: %[[INC_I:.+]] = add i64 %[[IV_I]], 1
; CHECK-NEXT: %[[CMP_I:.+]] = icmp eq i64 %[[INC_I]], %[[END]]
; CHECK-NEXT: br i1 %[[CMP_I]], label %[[EXIT_I:.+]], label %[[HEADER_I]],
; CHECK-SAME: !llvm.loop ![[LOOP_I:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT_I]]:
; CHECK-NEXT: %[[PARTIAL:.+]] = load i64, ptr %[[LOCAL]]
; CHECK-NEXT: atomicrmw add ptr %[[RESULT]], i64 %[[PARTIAL]] monotonic
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
; CHECK-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[LOOP_I]] = distinct !{![[LOOP_I]]}
; CHECK-DAG: ![[LOOP_O]] = distinct !{![[LOOP_O]], {{.+}}, ![[PREPARED]]}

declare void @sum(ptr %res, i64 %v)

define void @f(i64 %n) {
entry:
  %result = alloca i64
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %entry ], [ %inc.j, %for.j.latch ]
  detach within %syncreg, label %for.j.body, label %for.j.latch

for.j.body:
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1, i32 5, ptr %result, i64 %j, i32 0, i32 8, i64 0, ptr @sum)
  reattach within %syncreg, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !0

for.j.exit:
  sync within %syncreg, label %for.j.end

for.j.end:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"tapir.loop.reduction"}
!2 = !{!"tapir.loop.target", i32 1}
!3 = !{!"tapir.loop.perfect.depth", i32 1}
!4 = !{!"tapir.loop.perfect.level", i32 1}
