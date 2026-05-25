; Check that a simple reduction loop of depth 1 is prepared as expected.
;
; RUN: opt --tapir=hip --passes=kit-reductions -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK: %[[RESULT:.+]] = alloca i64
; CHECK: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK: %[[NREDS:.+]] = call i64 @llvm.kit.reduce.num.partials(i32 4, i64 %[[N]])
; CHECK-NEXT: %[[BYTES:.+]] = mul {{.+}} 4, %[[NREDS]]
; CHECK-NEXT: %[[REDS:.+]] = call {{.+}} @llvm.kit.mobile.alloc(i64 %[[BYTES]])
; CHECK-NEXT: call void {{.+}} @llvm.kit.mobile.init
; CHECK-SAME: i32 4
; CHECK-SAME: ptr {{[^%]+}} %[[REDS]]
; CHECK-SAME: i64 %[[NREDS]]
; CHECK-SAME: i32 [[UNIT:[^)]+]]
; CHECK-NEXT: br label %[[PH_O:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH_O]]:
; CHECK-NEXT: br label %[[HEADER_O:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_O]]:
; CHECK-NEXT: %[[IV_O:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH_O]] ],
; CHECK-SAME: [ %[[INC_O:.+]], %[[LATCH_O:.+]] ]
; CHECK-NEXT: detach within %[[SYNCREG]], label %[[GUARD_I:.+]], label %[[LATCH_O:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[GUARD_I]]:
; CHECK-NEXT: %[[CMP_GUARD:.+]] = icmp uge {{.+}} %[[IV_O]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP_GUARD]], label %[[END_I:.+]], label %[[PH_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH_I]]:
; CHECK-NEXT: br label %[[HEADER_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_I]]:
; CHECK-NEXT: %[[IV_I:.+]] = phi i64
; CHECK-SAME: [ %[[IV_O]], %[[PH_I]] ]
; CHECK-SAME: [ %[[INC_I:.+]], %[[LATCH_I:.+]] ]
; CHECK-NEXT: br label %[[BODY_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_I]]:
; CHECK-NEXT: %[[TRUNC:.+]] = trunc i64 %[[IV_I]] to i32
; CHECK-NEXT: %[[ADDR_DEST:.+]] = getelementptr {{.+}} %[[REDS]], i64 %[[IV_O]]
; CHECK-NEXT: %[[ADDR_CAST:.+]] = addrspacecast {{.+}} %[[ADDR_DEST]] to ptr
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0{{.*}}(
; CHECK-SAME: i32 4,
; CHECK-SAME: ptr %[[ADDR_CAST]],
; CHECK-SAME: i32 4,
; CHECK-SAME: i32 %[[TRUNC]],
; CHECK-SAME: i32 [[UNIT]],
; CHECK-SAME: ptr @mul)
; CHECK-NEXT: br label %[[LATCH_I]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_I]]:
; CHECK-NEXT: %[[INC_I:.+]] = add i64 %[[IV_I]], %[[NREDS]]
; CHECK-NEXT: %[[CMP_I:.+]] = icmp uge i64 %[[INC_I]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP_I]], label %[[EXIT_I:.+]], label %[[HEADER_I]],
; CHECK-SAME: !llvm.loop ![[LOOP_I:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT_I]]:
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
; CHECK-NEXT: br label %[[PARTIAL_REDUCE:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PARTIAL_REDUCE]]
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce.1{{[^(]*}}(
; CHECK-SAME: i32 4,
; CHECK-SAME: ptr %[[RESULT]],
; CHECK-SAME: i32 4,
; CHECK-SAME: ptr {{.+}} %[[REDS]],
; CHECK-SAME: i64 %[[NREDS]],
; CHECK-SAME: i32 [[UNIT]],
; CHECK-SAME: ptr @mul)
; CHECK-NEXT: br label %[[PARTIAL_FREE:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PARTIAL_FREE]]:
; CHECK-NEXT: call void @llvm.kit.mobile.free(ptr {{.+}} %[[REDS]])
; CHECK-NEXT: br label %[[SYNC:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC]]:
; CHECK-NEXT: sync within %[[SYNCREG]],
;
; CHECK-DAG: ![[TARGET:.+]] = !{!"tapir.loop.target", i32 4}
; CHECK-DAG: ![[REDUCTION:.+]] = !{!"tapir.loop.reduction"}
; CHECK-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.reduction.prepared"}
; CHECK-DAG: ![[LOOP_I]] = distinct !{![[LOOP_I]]}
; CHECK-DAG: ![[LOOP_O]] = distinct !{![[LOOP_O]], ![[REDUCTION]], ![[TARGET]], ![[PREPARED]]}

define void @mul(ptr %res, i32 %v) {
  %1 = load i32, ptr %res
  %2 = mul i32 %1, %v
  store i32 %2, ptr %res
  ret void
}

define void @f1(i64 %n) {
entry:
  %result = alloca i64
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  %trunc = trunc i64 %i to i32
  call void(i32, ptr, i32, i32, i32, ptr, ...) @llvm.kit.reduce.0(i32 4, ptr %result, i32 4, i32 %trunc, i32 1, ptr @mul)
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
!1 = !{!"tapir.loop.target", i32 4}
!2 = distinct !{!2, !0, !1}
