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
; ALL-SAME: i64 %[[N:[^)]+]]
; ALL-NEXT: [[ENTRY:.+]]:
; ALL-NEXT: %[[RESULT:.+]] = alloca i32
; ALL-NEXT: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; ALL-NEXT: br label %[[PH_WRAP:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[PH_WRAP]]:
; ALL-NEXT: %[[IPT:.+]] = call i64 @llvm.kit.gpu.ipt(i32 4, ptr null)
; ALL-NEXT: %[[IPT_MINUS_1:.+]] = sub i64 %[[IPT]], 1
; ALL-NEXT: %[[N_PLUS_IPT_MINUS_1:.+]] = add i64 %[[N]], %[[IPT_MINUS_1]]
; ALL-NEXT: %[[OTC:.+]] = udiv i64 %[[N_PLUS_IPT_MINUS_1]], %[[IPT]]
; GLOBAL-NEXT: %[[SHADOW:.+]] = tail call noalias ptr @llvm.kit.gpu.malloc(i32 4, i64 4)
; GLOBAL-NEXT: call void @llvm.kit.gpu.memcpy.htod(i32 4, ptr %[[SHADOW]], ptr %[[RESULT]], i64 4)
; UVM-NEXT: %[[SHADOWBUF:.+]] = call ptr addrspace(67) @llvm.kit.mobile.alloc(i32 4, i64 4)
; UVM-NEXT: %[[SHADOW:.+]] = addrspacecast ptr addrspace(67) %[[SHADOWBUF]] to ptr
; UVM-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[SHADOW]], ptr %[[RESULT]], i64 4, i1 false)
; ALL-NEXT: br label %[[HEADER_WRAP:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[HEADER_WRAP]]:
; ALL-NEXT: %[[IV_WRAP:.+]] = phi i64
; ALL-SAME: [ 0, %[[PH_WRAP]] ]
; ALL-SAME: [ %[[NEXTV_WRAP:.+]], %[[LATCH_WRAP:.+]] ]
; ALL-NEXT: detach within %[[SYNCREG]], label %[[BODY_WRAP:.+]], label %[[LATCH_WRAP]]
; ALL-EMPTY:
; ALL-NEXT: [[BODY_WRAP]]:
; ALL-NEXT: %[[THRDS_PLUS_N:.+]] = add i64 %[[N]], %[[OTC]]
; ALL-NEXT: %[[THRDS_PLUS_N_1:.+]] = sub i64 %[[THRDS_PLUS_N]], 1
; ALL-NEXT: %[[PER_THRD:.+]] = udiv i64 %[[THRDS_PLUS_N_1]], %[[OTC]]
; ALL-NEXT: %[[START:.+]] = mul i64 %[[IV_WRAP]], %[[PER_THRD]]
; ALL-NEXT: %[[ENDMAX:.+]] = add i64 %[[START]], %[[PER_THRD]]
; ALL-NEXT: %[[STOP:.+]] = call i64 @llvm.umin.i64(i64 %[[ENDMAX]], i64 %[[N]])
; ALL-NEXT: %[[GUARD:.+]] = icmp uge i64 %[[START]], %[[STOP]]
; ALL-NEXT: br i1 %[[GUARD]], label %[[END:.+]], label %[[PH:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[PH]]:
; ALL-NEXT: br label %[[HEADER:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[HEADER]]:
; ALL-NEXT: %[[I:.+]] = phi i64
; ALL-SAME: [ %[[START]], %[[PH]] ]
; ALL-SAME: [ %[[NEXT:.+]], %[[LATCH:.+]] ]
; ALL-NEXT: br label %[[BODY:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[BODY]]:
; ALL-NEXT: %[[TRUNC:.+]] = trunc i64 %[[I]] to i32
; ALL-NEXT: call {{.+}} @llvm.kit.reduce
; ALL-SAME: i32 4
; ALL-SAME: i32 5
; ALL-SAME: ptr %[[SHADOW]]
; ALL-SAME: i32 %[[TRUNC]]
; ALL-SAME: i32 0
; ALL-SAME: i32 4
; ALL-SAME: i32 0
; ALL-SAME: ptr @sum
; ALL-NEXT: br label %[[LATCH]]
; ALL-EMPTY:
; ALL-NEXT: [[LATCH]]:
; ALL-NEXT: %[[NEXT:.+]] = add i64 %[[I]], 1
; ALL-NEXT: %[[CMP:.+]] = icmp eq i64 %[[NEXT]], %[[STOP]]
; ALL-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]]
; ALL-SAME: !llvm.loop ![[LOOP:[0-9]+]]
; ALL-EMPTY:
; ALL-NEXT: [[EXIT]]:
; ALL-NEXT: br label %[[END]]
; ALL-EMPTY:
; ALL-NEXT: [[END]]:
; ALL-NEXT: br label %[[REATTACH_WRAP:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[REATTACH_WRAP]]:
; ALL-NEXT: reattach within %[[SYNCREG]], label %[[LATCH_WRAP]]
; ALL-EMPTY:
; ALL-NEXT: [[LATCH_WRAP]]:
; ALL-NEXT: %[[NEXTV_WRAP]] = add i64 %[[IV_WRAP]], 1
; ALL-NEXT: %[[CMP_WRAP:.+]] = icmp eq i64 %[[NEXTV_WRAP]], %[[OTC]]
; ALL-NEXT: br i1 %[[CMP_WRAP]], label %[[EXIT_WRAP:.+]], label %[[HEADER_WRAP]]
; ALL-SAME: !llvm.loop ![[LOOP_WRAP:[0-9]+]]
; ALL-EMPTY:
; ALL-NEXT: [[EXIT_WRAP]]
; GLOBAL-NEXT: call void @llvm.kit.gpu.memcpy.dtoh(i32 4, ptr %[[RESULT]], ptr %[[SHADOW]], i64 4)
; GLOBAL-NEXT: call void @llvm.kit.gpu.free(i32 4, ptr %[[SHADOW]])
; UVM-NEXT: call void @llvm.memcpy.inline{{.+}}(ptr %[[RESULT]], ptr %[[SHADOW]], i64 4, i1 false)
; UVM-NEXT: %[[SHADOWBUF:.+]] = addrspacecast ptr %[[SHADOW]] to ptr addrspace(67)
; UVM-NEXT: call void @llvm.kit.mobile.free(i32 4, ptr addrspace(67) %[[SHADOWBUF]])
; ALL-NEXT: br label %[[END_WRAP:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[END_WRAP]]:
; ALL-NEXT: sync within %[[SYNCREG]], label %[[EXIT:.+]]
; ALL-EMPTY:
; ALL-NEXT: [[EXIT]]:
; ALL-NEXT: ret void
;
; ALL-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; ALL-DAG: ![[LOOP_WRAP]] = distinct !{![[LOOP_WRAP]], {{.+}}, ![[PREPARED]]}
; ALL-DAG: ![[LOOP]] = distinct !{![[LOOP]]}

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
  call void(i32, i32, ptr, i32, i32, i32, i32, ptr, ...) @llvm.kit.reduce(i32 4, i32 5, ptr %result, i32 %trunc, i32 0, i32 4, i32 0, ptr @sum)
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
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.perfect.depth", i32 1}
!4 = !{!"tapir.loop.perfect.level", i32 1}
