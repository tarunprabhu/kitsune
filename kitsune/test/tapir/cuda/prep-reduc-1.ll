; Check that a simple reduction loop of depth 1 is prepared as expected.
;
; By default, the WarpShuffleOnly strategy is used for GPU reductions.
;
; RUN: opt --passes=kit-prepare -S %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,WSHF
;
; Otherwise, a reduce mode can be specified explicitly.
;
; RUN: opt --passes=kit-prepare --tapir-gpu-reduce-mode=direct -S %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,DIRECT
;
; RUN: opt --passes=kit-prepare --tapir-gpu-reduce-mode=mem -S %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,MEM
;
; RUN: opt --passes=kit-prepare --tapir-gpu-reduce-mode=wshf -S %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,WSHF
;
; RUN: opt --passes=kit-prepare --tapir-gpu-reduce-mode=wshfmem -S %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,WSHFMEM
;
; ------------------------------------------------------------------------------

; ALL-LABEL: @f
; ALL-SAME: i64 %[[N:[^)]+]]
; ALL: [[ENTRY:.+]]:
; ALL: %[[RESULT:.+]] = alloca i64
; ALL: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; ALL-NEXT: %[[GLOBAL:.+]] = tail call noalias ptr @llvm.kit.gpu.malloc(i32 2, i64 8)
; ALL-NEXT: %[[INIT:.+]] = load i64, ptr %[[RESULT]]
; ALL-NEXT: call {{.+}} @llvm.kit.gpu.memset.i64(i32 2, ptr %[[GLOBAL]], i64 1, i64 %[[INIT]])
; ALL-NEXT: br label %[[HEADER:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[HEADER]]:
; ALL-NEXT: %[[IV:.+]] = phi i64
; ALL-SAME: [ 0, %[[ENTRY]] ],
; ALL-SAME: [ %[[INC:.+]], %[[LATCH:.+]] ]
; ALL-NEXT: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
; ALL-EMPTY:
; ALL-NEXT: [[BODY]]:
; ALL-NEXT: %[[LOCAL:.+]] = alloca [8 x i8]
; ALL-NEXT: store i64 0, ptr %[[LOCAL]]
; ALL-NEXT: call {{.+}} @llvm.kit.reduce.0
; ALL-SAME: i32 2
; ALL-SAME: i32 5
; ALL-SAME: ptr %[[LOCAL]]
; ALL-SAME: i32 8
; ALL-SAME: i64 %[[IV]]
; ALL-SAME: i64 0
; ALL-SAME: ptr @sum
; ALL-NEXT: %[[VALUE:.+]] = load i64, ptr %[[LOCAL]]
; DIRECT-NEXT: call {{.+}} @llvm.kit.gpu.reduce.direct
; MEM-NEXT: call {{.+}} @llvm.kit.gpu.reduce.shared.memory
; WSHF-NEXT: call {{.+}} @llvm.kit.gpu.reduce.warp.shuffle
; WSHFMEM-NEXT: call {{.+}} @llvm.kit.gpu.reduce.warp.shuffle.shared.memory
; ALL-SAME: i32 2
; ALL-SAME: i32 5
; ALL-SAME: ptr %[[GLOBAL]]
; ALL-SAME: i32 8
; ALL-SAME: i64 %[[VALUE]]
; ALL-SAME: i64 0
; ALL-SAME: ptr @sum
; ALL-NEXT: reattach within %[[SYNCREG]], label %[[LATCH]]
; ALL-EMPTY:
; ALL-NEXT: [[LATCH]]:
; ALL-NEXT: %[[INC:.+]] = add i64 %[[IV]], 1
; ALL-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
; ALL-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]],
; ALL-SAME: !llvm.loop ![[LOOP:[0-9]+]]
; ALL-EMPTY:
; ALL-NEXT: [[EXIT]]:
; ALL-NEXT: call void @llvm.kit.gpu.memcpy.dtoh(i32 2, ptr %[[RESULT]], ptr %[[GLOBAL]], i64 8)
; ALL-NEXT: call void @llvm.kit.gpu.free(i32 2, ptr %[[GLOBAL]])
; ALL-NEXT: sync within %[[SYNCREG]],
;
; ALL-DAG: ![[TARGET:.+]] = !{!"tapir.loop.target", i32 2}
; ALL-DAG: ![[REDUCTION:.+]] = !{!"tapir.loop.reduction"}
; ALL-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; ALL-DAG: ![[LOOP]] = distinct !{![[LOOP]], ![[REDUCTION]], ![[TARGET]], ![[PREPARED]]}

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
