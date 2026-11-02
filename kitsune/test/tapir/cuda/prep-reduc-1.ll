; Check that a simple reduction loop of depth 1 is prepared as expected.
;
; By default, shadow memory is allocated using UVM.
;
; RUN: opt --passes=kit-prepare -S %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,UVM
;
; ------------------------------------------------------------------------------
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
; ALL-SAME: ptr %[[RESULT:[^,]+]]
; ALL-SAME: i64 %[[N1:[^)]+]]
; ALL-NEXT: [[ENTRY:.+]]:
; ALL-NEXT: %[[SR1:.+]] = tail call token @llvm.syncregion.start()
; ALL-NEXT: %[[CMPN1:.+]] = icmp
; ALL-NEXT: br i1 %[[CMPN1]], label %[[PH1:.+]], label %[[SYNC1:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[PH1]]:
; GLOBAL-NEXT: %[[SHADOW:.+]] = tail call noalias ptr @llvm.kit.gpu.malloc(i32 2, i64 8)
; GLOBAL-NEXT: call void @llvm.kit.gpu.memcpy.htod(i32 2, ptr %[[SHADOW]], ptr %[[RESULT]], i64 8)
; UVM-NEXT: %[[SHADOWBUF:.+]] = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 2, i64 8)
; UVM-NEXT: %[[SHADOW:.+]] = addrspacecast ptr addrspace(67) %[[SHADOWBUF]] to ptr
; UVM-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[SHADOW]], ptr %[[RESULT]], i64 8, i1 false)
; ALL-NEXT: br label %[[HEADER1:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[HEADER1]]:
; ALL-NEXT: %[[I1:.+]] = phi i64
; ALL-SAME: [ 0, %[[PH1]] ]
; ALL-SAME: [ %[[INC1:.+]], %[[LATCH1:.+]] ]
; ALL-NEXT: detach within %[[SR1]], label %[[BODY1:.+]], label %[[LATCH1]]
; ALL-EMPTY:
; ALL-NEXT: [[BODY1]]:
; ALL-NEXT: call {{.+}} @llvm.kit.reduce
; ALL-SAME: i32 2
; ALL-SAME: i32 5
; ALL-SAME: ptr %[[SHADOW]]
; ALL-SAME: i64 %[[I1]]
; ALL-SAME: i32 0
; ALL-SAME: i32 8
; ALL-SAME: i64 0
; ALL-SAME: ptr @sum
; ALL-NEXT: reattach within %[[SR1]], label %[[LATCH1]]
; ALL-EMPTY:
; ALL-NEXT: [[LATCH1]]:
; ALL-NEXT: %[[INC1:.+]] = add i64 %[[I1]], 1
; ALL-NEXT: %[[CMP1:.+]] = icmp eq i64 %[[INC1]], %[[N1]]
; ALL-NEXT: br i1 %[[CMP1]], label %[[LOOPEXIT1:.+]], label %[[HEADER1]]
; ALL-SAME: !llvm.loop ![[LOOP1:[0-9]+]]
; ALL-EMPTY:
; ALL-NEXT: [[LOOPEXIT1]]:
; GLOBAL-NEXT: call void @llvm.kit.gpu.memcpy.dtoh(i32 2, ptr %[[RESULT]], ptr %[[SHADOW]], i64 8)
; GLOBAL-NEXT: call void @llvm.kit.gpu.free(i32 2, ptr %[[SHADOW]])
; UVM-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[RESULT]], ptr %[[SHADOW]], i64 8, i1 false)
; UVM-NEXT: %[[SHADOWBUF:.+]] = addrspacecast ptr %[[SHADOW]] to ptr addrspace(67)
; UVM-NEXT: call void @llvm.kit.mobile.free(i32 2, ptr addrspace(67) %[[SHADOWBUF]])
; ALL-NEXT: br label %[[SYNC1]]
; ALL-EMPTY:
; ALL-NEXT: [[SYNC1]]
; ALL-NEXT: sync within %[[SR1]], label %[[EXIT:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[EXIT]]:
; ALL-NEXT: ret void
;
; ALL-DAG: ![[DEPTH:.+]] = !{!"tapir.loop.perfect.depth", i32 1}
; ALL-DAG: ![[L1:.+]] = !{!"tapir.loop.perfect.level", i32 1}
; ALL-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; ALL-DAG: ![[REDUCTION:.+]] = !{!"tapir.loop.reduction"}
; ALL-DAG: ![[TARGET:.+]] = !{!"tapir.loop.target", i32 2}

; ALL-DAG: ![[LOOP1]] = distinct !{![[LOOP1]], ![[TARGET]], ![[DEPTH]], ![[L1]], ![[REDUCTION]], ![[PREPARED]]}

declare void @sum(ptr, i64)

define void @f1(ptr %result, i64 %n1) {
entry:
  %sr1 = tail call token @llvm.syncregion.start()
  %cmpn1 = icmp ult i64 0, %n1
  br i1 %cmpn1, label %ph1, label %sync1

ph1:
  br label %header1

header1:
  %i1 = phi i64 [ 0, %ph1 ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 2, i32 5, ptr %result, i64 %i1, i32 0, i32 8, i64 0, ptr @sum)
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp.i = icmp eq i64 %inc1, %n1
  br i1 %cmp.i, label %loopexit1, label %header1, !llvm.loop !4

loopexit1:
  br label %sync1

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 2}
!1 = !{!"tapir.loop.perfect.depth", i32 1}
!2 = !{!"tapir.loop.perfect.level", i32 1}
!3 = !{!"tapir.loop.reduction"}
!4 = distinct !{!4, !0, !1, !2, !3}
