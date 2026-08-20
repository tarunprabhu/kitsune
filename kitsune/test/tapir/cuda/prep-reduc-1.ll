; Check that a simple reduction loop of depth 1 is prepared as expected.
;
; RUN: opt --tapir=cuda --passes=kit-prepare -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK: [[ENTRY:.+]]:
; CHECK: %[[RESULT:.+]] = alloca i64
; CHECK: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK-NEXT: %[[LOCAL:.+]] = tail call noalias ptr @llvm.kit.gpu.malloc(i32 2, i64 8)
; CHECK-NEXT: store i64 0, ptr %[[LOCAL]]
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[IV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ],
; CHECK-SAME: [ %[[INC:.+]], %[[LATCH:.+]] ]
; CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: atomicrmw add ptr %[[LOCAL]], i64 %[[IV]] monotonic
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[IV]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]],
; CHECK-SAME: !llvm.loop ![[LOOP:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: call void @llvm.kit.gpu.memcpy.dtoh(i32 2, ptr %[[RESULT]], ptr %[[LOCAL]], i64 8)
; CHECK-NEXT: call void @llvm.kit.gpu.free(i32 2, ptr %[[LOCAL]])
; CHECK-NEXT: sync within %[[SYNCREG]],
;
; CHECK-DAG: ![[TARGET:.+]] = !{!"tapir.loop.target", i32 2}
; CHECK-DAG: ![[REDUCTION:.+]] = !{!"tapir.loop.reduction"}
; CHECK-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[LOOP]] = distinct !{![[LOOP]], ![[REDUCTION]], ![[TARGET]], ![[PREPARED]]}

declare void @sum (ptr %res, i64 %v)

define void @f1(i64 %n) {
entry:
  %result = alloca i64
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  call void(i32, i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 2, i32 5, ptr %result, i32 8, i64 %i, i64 0, ptr @sum)
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
!1 = !{!"tapir.loop.target", i32 2}
!2 = distinct !{!2, !0, !1}
