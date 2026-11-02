; Check that multiple reductions in a loop are handled correctly.
;
; RUN: opt --passes=kit-prepare -S %s | FileCheck %s
;
; ------------------------------------------------------------------------------

; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[SUM:[^,]+]]
; CHECK-SAME: ptr %[[UMAX:[^,]+]]
; CHECK-SAME: i64 %[[N1:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SR1:.+]] = tail call token @llvm.syncregion.start()
; CHECK-NEXT: %[[CMPN1:.+]] = icmp ult i64 0, %[[N1]]
; CHECK-NEXT: br i1 %[[CMPN1]], label %[[PH1:.+]], label %[[SYNC1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH1]]:
; CHECK-NEXT: %[[SHADOWBUF1:.+]] = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 4, i64 8)
; CHECK-NEXT: %[[SHADOW1:.+]] = addrspacecast ptr addrspace(67) %[[SHADOWBUF1]] to ptr
; CHECK-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[SHADOW1]], ptr %[[SUM]], i64 8, i1 false)
; CHECK-NEXT: %[[SHADOWBUF2:.+]] = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 4, i64 8)
; CHECK-NEXT: %[[SHADOW2:.+]] = addrspacecast ptr addrspace(67) %[[SHADOWBUF2]] to ptr
; CHECK-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[SHADOW2]], ptr %[[UMAX]], i64 8, i1 false)
; CHECK-NEXT: br label %[[HEADER1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER1]]:
; CHECK-NEXT: %[[I1:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH1]] ]
; CHECK-SAME: [ %[[INC1:.+]], %[[LATCH1:.+]] ]
; CHECK-NEXT: detach within %[[SR1]], label %[[BODY1:.+]], label %[[LATCH1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY1]]:
; CHECK-NEXT: %[[ODD:.+]] = and i64 %[[I1]], 1
; CHECK-NEXT: %[[CMPO:.+]] = icmp eq i64 %[[ODD]], 1
; CHECK-NEXT: br i1 %[[CMPO]], label %[[BODY1O:.+]], label %[[REATTACH1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY1O]]:
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 4
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[SHADOW1]]
; CHECK-SAME: i64 %[[I1]]
; CHECK-SAME: i32 0
; CHECK-SAME: i32 8
; CHECK-SAME: i64 0
; CHECK-SAME: ptr @sum
; CHECK-NEXT: br label %[[REATTACH1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH1]]:
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 4
; CHECK-SAME: i32 26
; CHECK-SAME: ptr %[[SHADOW2]]
; CHECK-SAME: i64 %[[I1]]
; CHECK-SAME: i32 0
; CHECK-SAME: i32 8
; CHECK-SAME: i64 0
; CHECK-SAME: ptr @umax
; CHECK-NEXT: reattach within %[[SR1]], label %[[LATCH1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH1]]:
; CHECK-NEXT: %[[INC1:.+]] = add i64 %[[I1]], 1
; CHECK-NEXT: %[[CMP1:.+]] = icmp eq i64 %[[INC1]], %[[N1]]
; CHECK-NEXT: br i1 %[[CMP1]], label %[[LOOPEXIT1:.+]], label %[[HEADER1]]
; CHECK-SAME: !llvm.loop ![[LOOP1:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT1]]:
; CHECK-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[SUM]], ptr %[[SHADOW1]], i64 8, i1 false)
; CHECK-NEXT: %[[SHADOWBUF1:.+]] = addrspacecast ptr %[[SHADOW1]] to ptr addrspace(67)
; CHECK-NEXT: call void @llvm.kit.mobile.free(i32 4, ptr addrspace(67) %[[SHADOWBUF1]])
; CHECK-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[UMAX]], ptr %[[SHADOW2]], i64 8, i1 false)
; CHECK-NEXT: %[[SHADOWBUF2:.+]] = addrspacecast ptr %[[SHADOW2]] to ptr addrspace(67)
; CHECK-NEXT: call void @llvm.kit.mobile.free(i32 4, ptr addrspace(67) %[[SHADOWBUF2]])
; CHECK-NEXT: br label %[[SYNC1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC1]]
; CHECK-NEXT: sync within %[[SR1]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
;
; CHECK-DAG: ![[DEPTH:.+]] = !{!"tapir.loop.perfect.depth", i32 1}
; CHECK-DAG: ![[L1:.+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[REDUCTION:.+]] = !{!"tapir.loop.reduction"}
; CHECK-DAG: ![[TARGET:.+]] = !{!"tapir.loop.target", i32 4}

; CHECK-DAG: ![[LOOP1]] = distinct !{![[LOOP1]], ![[REDUCTION]], ![[TARGET]], ![[DEPTH]], ![[L1]], ![[PREPARED]]}

declare void @sum(ptr %res, i64 %v)
declare void @umax(ptr %res, i64 %v)

define void @f(ptr %sum, ptr %umax, i64 %n1) {
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
  %odd = and i64 %i1, 1
  %cmp.odd = icmp eq i64 %odd, 1
  br i1 %cmp.odd, label %body1o, label %reattach1

body1o:
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 4, i32 5, ptr %sum, i64 %i1, i32 0, i32 8, i64 0, ptr @sum)
  br label %reattach1

reattach1:
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 4, i32 26, ptr %umax, i64 %i1, i32 0, i32 8, i64 0, ptr @umax)
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n1
  br i1 %cmp1, label %loopexit1, label %header1, !llvm.loop !0

loopexit1:
  br label %sync1

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"tapir.loop.reduction"}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.perfect.depth", i32 1}
!4 = !{!"tapir.loop.perfect.level", i32 1}
