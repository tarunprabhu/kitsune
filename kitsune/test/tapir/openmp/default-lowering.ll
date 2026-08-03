; RUN: opt --tapir=openmp -passes='loop-spawning' %s -S | FileCheck %s
;
; CHECK: define {{.+}} @f(
; CHECK-SAME: ptr %[[A:[^,]+]],
; CHECK-SAME: i64 %[[N:[^)]+]])
; CHECK: call void (i32, ptr, i64, i64, ...) @llvm.kit.cpu.threads.launch
; CHECK-SAME: i32 512,
; CHECK-SAME: ptr @[[WRAPPER:[^,]+]],
; CHECK-SAME: i64 0,
; CHECK-SAME: i64 %[[N]],
; CHECK-SAME: ptr %[[A]])
;
; CHECK: define internal fastcc void @[[WRAPPER]](
; CHECK-SAME: i64 {{[^%]*}}%[[BEG:[^,]+]],
; CHECK-SAME: i64 {{[^%]*}}%[[END:[^,]+]],
; CHECK-SAME: ptr {{[^%]*}}%[[ARGS:[^)]+]])
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[OFF0:.+]] = getelementptr inbounds { ptr }, ptr %[[ARGS]], i32 0, i32 0
; CHECK-NEXT: %[[A:.+]] = load ptr, ptr %[[OFF0]]
; CHECK-NEXT: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOP:.+]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ %[[BEG]], %[[ENTRY]] ]
; CHECK-SAME: [ %[[NEXT:.+]], %[[LATCH:.+]] ]
; CHECK-NEXT: %[[IDX:.+]] = getelementptr
; CHECK-NEXT: store i64 %[[I]], ptr %[[IDX]]
; CHECK-NEXT: %[[NEXT]] = add i64 %[[I]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[NEXT]], %[[END]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[LOOP]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: br label %[[END:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END]]:
; CHECK-NEXT: ret void
;
; CHECK-DAG: !{!"kit.module.tts", ![[TTS:[0-9]+]]}
; CHECK-DAG: !{i32 512}

define void @f(ptr %a, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %next.i, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %idx = getelementptr i64, ptr %a, i64 %i
  store i64 %i, ptr %idx
  reattach within %syncreg, label %latch

latch:
  %next.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %next.i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !0

exit:
  sync within %syncreg, label %end

end:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.target", i32 512}
!2 = !{!"tapir.loop.spawn.strategy", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}
