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
; CHECK-NEXT: br label %[[PH_WRAP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH_WRAP]]:
; CHECK-NEXT: %[[IPT:.+]] = call i64 @llvm.kit.gpu.ipt(i32 2, ptr null)
; CHECK-NEXT: %[[IPT_MINUS_1:.+]] = sub i64 %[[IPT]], 1
; CHECK-NEXT: %[[N_PLUS_IPT_MINUS_1:.+]] = add i64 %[[N]], %[[IPT_MINUS_1]]
; CHECK-NEXT: %[[OTC:.+]] = udiv i64 %[[N_PLUS_IPT_MINUS_1]], %[[IPT]]
; CHECK-NEXT: %[[BUF1:.+]] = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 2, i64 8)
; CHECK-NEXT: %[[SHADOW1:.+]] = addrspacecast ptr addrspace(67) %[[BUF1]] to ptr
; CHECK-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[SHADOW1]], ptr %[[SUM]], i64 8, i1 false)
; CHECK-NEXT: %[[BUF2:.+]] = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 2, i64 8)
; CHECK-NEXT: %[[SHADOW2:.+]] = addrspacecast ptr addrspace(67) %[[BUF2]] to ptr
; CHECK-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[SHADOW2]], ptr %[[UMAX]], i64 8, i1 false)
; CHECK-NEXT: br label %[[HEADER_WRAP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER_WRAP]]:
; CHECK-NEXT: %[[IV_WRAP:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH_WRAP]] ]
; CHECK-SAME: [ %[[NEXTV_WRAP:.+]], %[[LATCH_WRAP:.+]] ]
; CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY_WRAP:.+]], label %[[LATCH_WRAP]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY_WRAP]]:
; CHECK-NEXT: %[[THRDS_PLUS_N:.+]] = add i64 %[[N]], %[[OTC]]
; CHECK-NEXT: %[[THRDS_PLUS_N_1:.+]] = sub i64 %[[THRDS_PLUS_N]], 1
; CHECK-NEXT: %[[PER_THRD:.+]] = udiv i64 %[[THRDS_PLUS_N_1]], %[[OTC]]
; CHECK-NEXT: %[[START:.+]] = mul i64 %[[IV_WRAP]], %[[PER_THRD]]
; CHECK-NEXT: %[[ENDMAX:.+]] = add i64 %[[START]], %[[PER_THRD]]
; CHECK-NEXT: %[[STOP:.+]] = call i64 @llvm.umin.i64(i64 %[[ENDMAX]], i64 %[[N]])
; CHECK-NEXT: %[[GUARD:.+]] = icmp uge i64 %[[START]], %[[STOP]]
; CHECK-NEXT: br i1 %[[GUARD]], label %[[END:.+]], label %[[PH:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH]]:
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ %[[START]], %[[PH]] ]
; CHECK-SAME: [ %[[NEXT:.+]], %[[LATCH:.+]] ]
; CHECK-NEXT: br label %[[BODY:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: %[[ODD:.+]] = and i64 %i, 1
; CHECK-NEXT: %[[ISODD:.+]] = icmp eq i64 %[[ODD]], 1
; CHECK-NEXT: br i1 %[[ISODD]], label %[[REDUCE:.+]], label %[[REATTACH:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REDUCE]]:
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 2
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[SHADOW1]]
; CHECK-SAME: i64 %[[I]]
; CHECK-SAME: i32 0
; CHECK-SAME: i32 8
; CHECK-SAME: i64 0
; CHECK-SAME: ptr @sum
; CHECK-NEXT: br label %[[REATTACH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH]]:
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 2
; CHECK-SAME: i32 26
; CHECK-SAME: ptr %[[SHADOW2]]
; CHECK-SAME: i64 %[[I]]
; CHECK-SAME: i32 0
; CHECK-SAME: i32 8
; CHECK-SAME: i64 0
; CHECK-SAME: ptr @umax
; CHECK-NEXT: br label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[NEXT:.+]] = add i64 %[[I]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[NEXT]], %[[STOP]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-SAME: !llvm.loop ![[LOOP:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: br label %[[END]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END]]:
; CHECK-NEXT: br label %[[REATTACH_WRAP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH_WRAP]]:
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH_WRAP]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH_WRAP]]:
; CHECK-NEXT: %[[NEXTV_WRAP]] = add i64 %[[IV_WRAP]], 1
; CHECK-NEXT: %[[CMP_WRAP:.+]] = icmp eq i64 %[[NEXTV_WRAP]], %[[OTC]]
; CHECK-NEXT: br i1 %[[CMP_WRAP]], label %[[EXIT_WRAP:.+]], label %[[HEADER_WRAP]]
; CHECK-SAME: !llvm.loop ![[LOOP_WRAP:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT_WRAP]]
; CHECK-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[SUM]], ptr %[[SHADOW1]], i64 8, i1 false)
; CHECK-NEXT: %[[BUF1:.+]] = addrspacecast ptr %[[SHADOW1]] to ptr addrspace(67)
; CHECK-NEXT: call void @llvm.kit.mobile.free(i32 2, ptr addrspace(67) %[[BUF1]])
; CHECK-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[UMAX]], ptr %[[SHADOW2]], i64 8, i1 false)
; CHECK-NEXT: %[[BUF2:.+]] = addrspacecast ptr %[[SHADOW2]] to ptr addrspace(67)
; CHECK-NEXT: call void @llvm.kit.mobile.free(i32 2, ptr addrspace(67) %[[BUF2]])
; CHECK-NEXT: br label %[[END_WRAP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END_WRAP]]:
; CHECK-NEXT: sync within %[[SYNCREG]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
;
; CHECK-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[LOOP_WRAP]] = distinct !{![[LOOP_WRAP]], {{.+}}, ![[PREPARED]]}
; CHECK-DAG: ![[LOOP]] = distinct !{![[LOOP]]}

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
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 2, i32 5, ptr %sum, i64 %i, i32 0, i32 8, i64 0, ptr @sum)
  br label %reattach

reattach:
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 2, i32 26, ptr %umax, i64 %i, i32 0, i32 8, i64 0, ptr @umax)
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg, label %for.i.end

for.i.end:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"tapir.loop.reduction"}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.perfect.depth", i32 1}
!4 = !{!"tapir.loop.perfect.level", i32 1}
