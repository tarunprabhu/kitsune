; Check that a simple reduction loop of depth 1 is prepared as expected.
;
; By default, shadow memory is allocated using UVM.
;
; RUN: opt --passes=kit-prepare -S %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,UVM
;
; Otherwise, the shadow memory to use can be specified explicitly.
;
; RUN: opt --passes=kit-prepare --tapir-gpu-reduce-shadow=global -S %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,GLOBAL
;
; RUN: opt --passes=kit-prepare --tapir-gpu-reduce-shadow=uvm -S %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,UVM
;
; ------------------------------------------------------------------------------

; ALL-LABEL: @f
; ALL-SAME: i64 %[[N:[^)]+]]
; ALL: [[ENTRY:.+]]:
; ALL: %[[RESULT:.+]] = alloca i32
; ALL: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; GLOBAL-NEXT: %[[SHADOW:.+]] = tail call noalias ptr @llvm.kit.gpu.malloc(i32 4, i64 4)
; GLOBAL-NEXT: call void @llvm.kit.gpu.memcpy.htod(i32 4, ptr %[[SHADOW]], ptr %[[RESULT]], i64 4)
; UVM-NEXT: %[[SHADOWBUF:.+]] = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 4, i64 4)
; UVM-NEXT: %[[SHADOW:.+]] = addrspacecast ptr addrspace(67) %[[SHADOWBUF]] to ptr
; UVM-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[SHADOW]], ptr %[[RESULT]], i64 4, i1 false)
; ALL-NEXT: br label %[[HEADER:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[HEADER]]:
; ALL-NEXT: %[[IV:.+]] = phi i64
; ALL-SAME: [ 0, %[[ENTRY]] ],
; ALL-SAME: [ %[[INC:.+]], %[[LATCH:.+]] ]
; ALL-NEXT: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
; ALL-EMPTY:
; ALL-NEXT: [[BODY]]:
; ALL-NEXT: %[[TRUNC:.+]] = trunc i64 %[[IV]] to i32
; ALL-NEXT: call {{.+}} @llvm.kit.reduce.0
; ALL-SAME: i32 4
; ALL-SAME: i32 5
; ALL-SAME: ptr %[[SHADOW]]
; ALL-SAME: i32 4
; ALL-SAME: i32 %[[TRUNC]]
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
; GLOBAL-NEXT: call void @llvm.kit.gpu.memcpy.dtoh(i32 4, ptr %[[RESULT]], ptr %[[SHADOW]], i64 4)
; GLOBAL-NEXT: call void @llvm.kit.gpu.free(i32 4, ptr %[[SHADOW]])
; UVM-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[RESULT]], ptr %[[SHADOW]], i64 4, i1 false)
; UVM-NEXT: %[[SHADOWBUF:.+]] = addrspacecast ptr %[[SHADOW]] to ptr addrspace(67)
; UVM-NEXT: call void @llvm.kit.mobile.free(i32 4, ptr addrspace(67) %[[SHADOWBUF]])
; ALL-NEXT: sync within %[[SYNCREG]],
;
; ALL-DAG: ![[TARGET:.+]] = !{!"tapir.loop.target", i32 4}
; ALL-DAG: ![[REDUCTION:.+]] = !{!"tapir.loop.reduction"}
; ALL-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; ALL-DAG: ![[LOOP]] = distinct !{![[LOOP]], ![[REDUCTION]], ![[TARGET]], ![[PREPARED]]}

declare void @sum(ptr %res, i32 %v)

define void @f1(i64 %n) {
entry:
  %result = alloca i32
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  %trunc = trunc i64 %i to i32
  call void(i32, i32, ptr, i32, i32, i32, ptr, ...) @llvm.kit.reduce.0(i32 4, i32 5, ptr %result, i32 4, i32 %trunc, i32 0, ptr @sum)
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
