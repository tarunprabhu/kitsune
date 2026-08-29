; The GPU reduction intrinsics should be added to the end of the loop, not
; immediately after the reduce intrinsic call.
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
; ALL-NEXT: %[[GLOBAL:.+]] = tail call noalias ptr @llvm.kit.gpu.malloc(i32 4, i64 4)
; ALL-NEXT: %[[INIT:.+]] = load i32, ptr %[[RESULT]]
; ALL-NEXT: call {{.+}} @llvm.kit.gpu.memset.i32(i32 4, ptr %[[GLOBAL]], i64 1, i32 %[[INIT]])
; ALL-NEXT: br label %[[HEADER:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[HEADER]]:
; ALL-NEXT: %[[IV:.+]] = phi i64
; ALL-SAME: [ 0, %[[ENTRY]] ],
; ALL-SAME: [ %[[INC:.+]], %[[LATCH:.+]] ]
; ALL-NEXT: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
; ALL-EMPTY:
; ALL-NEXT: [[BODY]]:
; ALL-NEXT: %[[LOCAL:.+]] = alloca [4 x i8]
; ALL-NEXT: store i32 0, ptr %[[LOCAL]]
; ALL-NEXT: %[[ODD:.+]] = and i64 %i, 1
; ALL-NEXT: %[[ISODD:.+]] = icmp eq i64 %[[ODD]], 1
; ALL-NEXT: br i1 %[[ISODD]], label %[[REDUCE:.+]], label %[[REATTACH:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[REDUCE]]:
; ALL-NEXT: %[[TRUNC:.+]] = trunc i64 %[[IV]] to i32
; ALL-NEXT: call {{.+}} @llvm.kit.reduce.0
; ALL-SAME: i32 4
; ALL-SAME: i32 5
; ALL-SAME: ptr %[[LOCAL]]
; ALL-SAME: i32 4
; ALL-SAME: i32 %[[TRUNC]]
; ALL-SAME: i32 0
; ALL-SAME: ptr @sum
; ALL-NEXT: br label %[[REATTACH]]
; ALL-EMPTY:
; ALL-NEXT: [[REATTACH]]:
; ALL-NEXT: %[[VALUE:.+]] = load i32, ptr %[[LOCAL]]
; DIRECT-NEXT: call {{.+}} @llvm.kit.gpu.reduce.direct
; MEM-NEXT: call {{.+}} @llvm.kit.gpu.reduce.shared.memory
; WSHF-NEXT: call {{.+}} @llvm.kit.gpu.reduce.warp.shuffle
; WSHFMEM-NEXT: call {{.+}} @llvm.kit.gpu.reduce.warp.shuffle.shared.memory
; ALL-SAME: i32 4
; ALL-SAME: i32 5
; ALL-SAME: ptr %[[GLOBAL]]
; ALL-SAME: i32 4
; ALL-SAME: i32 %[[VALUE]]
; ALL-SAME: i32 0
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
; ALL-NEXT: call void @llvm.kit.gpu.memcpy.dtoh(i32 4, ptr %[[RESULT]], ptr %[[GLOBAL]], i64 4)
; ALL-NEXT: call void @llvm.kit.gpu.free(i32 4, ptr %[[GLOBAL]])
; ALL-NEXT: sync within %[[SYNCREG]],
;
; ALL-DAG: ![[TARGET:.+]] = !{!"tapir.loop.target", i32 4}
; ALL-DAG: ![[REDUCTION:.+]] = !{!"tapir.loop.reduction"}
; ALL-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; ALL-DAG: ![[LOOP]] = distinct !{![[LOOP]], ![[REDUCTION]], ![[TARGET]], ![[PREPARED]]}

declare void @sum (ptr %res, i32 %v)

define void @f1(i64 %n) {
entry:
  %result = alloca i64
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  %odd = and i64 %i, 1
  %cmp.odd = icmp eq i64 %odd, 1
  br i1 %cmp.odd, label %reduce, label %reattach

reduce:
  %i.32 = trunc i64 %i to i32
  call void(i32, i32, ptr, i32, i32, i32, ptr, ...) @llvm.kit.reduce.0(i32 4, i32 5, ptr %result, i32 4, i32 %i.32, i32 0, ptr @sum)
  br label %reattach

reattach:
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
