; Check that a simple reduction loop of depth 1 is finalized as expected.
;
; By default, the WarpShuffleOnly strategy is used for GPU reductions.
;
; RUN: opt --tapir=hip --passes=loop-spawning,emb-finalize-reductions %s \
; RUN:     --tapir-gpu-reduce-mode=wshf \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s --check-prefixes=ALL,WSHF
;
; Otherwise, a reduce mode can be specified explicitly.
;
; RUN: opt --tapir=hip -passes=loop-spawning,emb-finalize-reductions %s \
; RUN:     --tapir-gpu-reduce-mode=direct \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s --check-prefixes=ALL,DIRECT
;
; RUN: opt --tapir=hip -passes=loop-spawning,emb-finalize-reductions %s \
; RUN:     --tapir-gpu-reduce-mode=mem \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s --check-prefixes=ALL,MEM
;
; RUN: opt --tapir=hip --passes=loop-spawning,emb-finalize-reductions %s\
; RUN:     --tapir-gpu-reduce-mode=wshf \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s --check-prefixes=ALL,WSHF
;
; RUN: opt --tapir=hip --passes=loop-spawning,emb-finalize-reductions %s \
; RUN:     --tapir-gpu-reduce-mode=wshfmem \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s --check-prefixes=ALL,WSHFMEM
;
; ------------------------------------------------------------------------------

; ALL-LABEL: define
; ALL-SAME: %[[ZERO:[^,]+]]
; ALL-SAME: %[[TC:[^,]+]]
; ALL-SAME: %[[RESULT:[^)]+]]
; ALL-NEXT: [[ENTRY:.+]]:
; ALL-NEXT: %[[LOCAL:.+]] = alloca i64, {{.+}}addrspace({{.+}})
; ALL-NEXT: store i64 0, ptr addrspace({{.+}}) %[[LOCAL]]
; ALL: br i1 %{{.+}}, label %[[HEADER:.+]], label %[[EXIT:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[HEADER]]:
; ALL-NEXT: %[[IV:.+]] = phi i64
; ALL-SAME: [ %{{.+}}, %[[ENTRY]] ],
; ALL-SAME: [ %[[INC:.+]], %[[LATCH:.+]] ]
; ALL-NEXT: br label %[[BODY:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[BODY]]:
; ALL-NEXT: %[[CST:.+]] = addrspacecast ptr addrspace({{.+}}) %[[LOCAL]] to ptr
; ALL-NEXT: call {{.+}} @llvm.kit.reduce.0
; ALL-SAME: i32 4
; ALL-SAME: i32 5
; ALL-SAME: ptr %[[CST]]
; ALL-SAME: i32 8
; ALL-SAME: i64 %[[IV]]
; ALL-SAME: i64 0
; ALL-SAME: ptr @sum
; ALL-NEXT: br label %[[LATCH]]
; ALL-EMPTY:
; ALL-NEXT: [[LATCH]]:
; ALL-NEXT: %[[INC:.+]] = add i64 %[[IV]], 1
; ALL-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %{{.+}}
; ALL-NEXT: br i1 %[[CMP]], label %[[EXIT]], label %[[HEADER]],
; ALL-SAME: !llvm.loop ![[LOOP:[0-9]+]]
; ALL-EMPTY:
; ALL-NEXT: [[EXIT]]:
; ALL-NEXT: %[[VALUE:.+]] = load i64, ptr addrspace({{.+}}) %[[LOCAL]]
; DIRECT-NEXT: call {{.+}} @llvm.kit.gpu.reduce.direct
; MEM-NEXT: call {{.+}} @llvm.kit.gpu.reduce.shared.memory
; WSHF-NEXT: call {{.+}} @llvm.kit.gpu.reduce.warp.shuffle
; WSHFMEM-NEXT: call {{.+}} @llvm.kit.gpu.reduce.warp.shuffle.shared.memory
; ALL-SAME: i32 4
; ALL-SAME: i32 5
; ALL-SAME: ptr %[[RESULT]]
; ALL-SAME: i32 8
; ALL-SAME: i64 %[[VALUE]]
; ALL-SAME: i64 0
; ALL-SAME: ptr @sum
; ALL-NEXT: ret void

declare void @sum (ptr %res, i64 %v)

define void @f1(ptr %result, i64 %n) {
entry:
  %syncreg = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  call void(i32, i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 4, i32 5, ptr %result, i32 8, i64 %i, i64 0, ptr @sum)
  reattach within %syncreg, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %exit, label %header, !llvm.loop !0

exit:
  sync within %syncreg, label %end

end:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5, !6}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}
!6 = !{!"tapir.loop.reduction"}
