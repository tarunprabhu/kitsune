; Check that multiple reductions in a loop are handled correctly.
;
; RUN: opt --passes=kit-prepare -S %s | FileCheck %s
;
; ------------------------------------------------------------------------------

; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N:[^,]+]]
; CHECK-SAME: ptr %[[SUM:[^,]+]]
; CHECK-SAME: ptr %[[UMAX:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK-NEXT: %[[BUF1:.+]] = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 4, i64 8)
; CHECK-NEXT: %[[SHADOW1:.+]] = addrspacecast ptr addrspace(67) %[[BUF1]] to ptr
; CHECK-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[SHADOW1]], ptr %[[SUM]], i64 8, i1 false)
; CHECK-NEXT: %[[BUF2:.+]] = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 4, i64 8)
; CHECK-NEXT: %[[SHADOW2:.+]] = addrspacecast ptr addrspace(67) %[[BUF2]] to ptr
; CHECK-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[SHADOW2]], ptr %[[UMAX]], i64 8, i1 false)
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[IV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ],
; CHECK-SAME: [ %[[INC:.+]], %[[LATCH:.+]] ]
; CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: %[[LOCAL2:.+]] = alloca [8 x i8]
; CHECK-NEXT: store i64 0, ptr %[[LOCAL2]]
; CHECK-NEXT: %[[LOCAL1:.+]] = alloca [8 x i8]
; CHECK-NEXT: store i64 0, ptr %[[LOCAL1]]
; CHECK-NEXT: %[[ODD:.+]] = and i64 %i, 1
; CHECK-NEXT: %[[ISODD:.+]] = icmp eq i64 %[[ODD]], 1
; CHECK-NEXT: br i1 %[[ISODD]], label %[[REDUCE:.+]], label %[[REATTACH:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REDUCE]]:
; CHECK-NEXT: call void @sum(ptr %[[LOCAL1]], i64 %[[IV]])
; CHECK-NEXT: br label %[[REATTACH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH]]:
; CHECK-NEXT: call void @umax(ptr %[[LOCAL2]], i64 %[[IV]])
; CHECK-NEXT: %[[TOKEN1:.+]] = call token @llvm.kit.gpu.convergent.begin(i32 4)
; CHECK-NEXT: %[[VALUE1:.+]] = load i64, ptr %[[LOCAL1]]
; CHECK-NEXT: call {{.+}} @llvm.kit.gpu.reduce.warp.shuffle
; CHECK-SAME: i32 4
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[SHADOW1]]
; CHECK-SAME: i32 8
; CHECK-SAME: i64 %[[VALUE1]]
; CHECK-SAME: i64 0
; CHECK-SAME: ptr @sum
; CHECK-NEXT: call void @llvm.kit.gpu.convergent.end(i32 4, token %[[TOKEN1]])
; CHECK-NEXT: %[[TOKEN2:.+]] = call token @llvm.kit.gpu.convergent.begin(i32 4)
; CHECK-NEXT: %[[VALUE2:.+]] = load i64, ptr %[[LOCAL2]]
; CHECK-NEXT: call {{.+}} @llvm.kit.gpu.reduce.warp.shuffle
; CHECK-SAME: i32 4
; CHECK-SAME: i32 26
; CHECK-SAME: ptr %[[SHADOW2]]
; CHECK-SAME: i32 8
; CHECK-SAME: i64 %[[VALUE2]]
; CHECK-SAME: i64 0
; CHECK-SAME: ptr @umax
; CHECK-NEXT: call void @llvm.kit.gpu.convergent.end(i32 4, token %[[TOKEN2]])
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[IV]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]],
; CHECK-SAME: !llvm.loop ![[LOOP:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[SUM]], ptr %[[SHADOW1]], i64 8, i1 false)
; CHECK-NEXT: %[[BUF1:.+]] = addrspacecast ptr %[[SHADOW1]] to ptr addrspace(67)
; CHECK-NEXT: call void @llvm.kit.mobile.free(i32 4, ptr addrspace(67) %[[BUF1]])
; CHECK-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[UMAX]], ptr %[[SHADOW2]], i64 8, i1 false)
; CHECK-NEXT: %[[BUF2:.+]] = addrspacecast ptr %[[SHADOW2]] to ptr addrspace(67)
; CHECK-NEXT: call void @llvm.kit.mobile.free(i32 4, ptr addrspace(67) %[[BUF2]])
; CHECK-NEXT: sync within %[[SYNCREG]],
;
; CHECK-DAG: ![[TARGET:.+]] = !{!"tapir.loop.target", i32 4}
; CHECK-DAG: ![[REDUCTION:.+]] = !{!"tapir.loop.reduction"}
; CHECK-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[LOOP]] = distinct !{![[LOOP]], ![[REDUCTION]], ![[TARGET]], ![[PREPARED]]}

declare void @sum(ptr %res, i64 %v)
declare void @umax(ptr %res, i64 %v)

define void @f(i64 %n, ptr %sum, ptr %umax) {
entry:
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
  call void(i32, i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 4, i32 5, ptr %sum, i32 8, i64 %i, i64 0, ptr @sum)
  br label %reattach

reattach:
  call void(i32, i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 4, i32 26, ptr %umax, i32 8, i64 %i, i64 0, ptr @umax)
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
