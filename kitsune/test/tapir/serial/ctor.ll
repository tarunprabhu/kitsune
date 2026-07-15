; Check that the global ctor calls the appropriate functions in Kitsune's
; runtime depending on the command line arguments passed.
;
; RUN: opt --tapir=serial -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     | FileCheck %s -check-prefix DEFAULT
;
; DEFAULT-LABEL: @llvm.global_ctors = appending global
; DEFAULT-SAME: { i32 65535, ptr @[[CTOR:.+]], ptr null }
;
; DEFAULT-LABEL: @llvm.global_dtors = appending global
; DEFAULT-SAME: { i32 65535, ptr @[[DTOR:.+]], ptr null }
;
; DEFAULT: define internal void @[[CTOR]]()
; DEFAULT-NEXT: [[ENTRY:.+]]:
; DEFAULT-NEXT: call {{.+}} @llvm.kit.runtime.initialize(i32 1)
; DEFAULT-NEXT: br label %[[EXIT:.+]]
; DEFAULT-EMPTY:
; DEFAULT-NEXT: [[EXIT]]:
; DEFAULT-NEXT: ret void
; DEFAULT-NEXT: }
;
; DEFAULT: define internal void @[[DTOR]]()
; DEFAULT-NEXT: [[ENTRY:.+]]:
; DEFAULT-NEXT: call {{.+}} @llvm.kit.runtime.finalize(i32 1)
; DEFAULT-NEXT: br label %[[EXIT:.+]]
; DEFAULT-EMPTY:
; DEFAULT-NEXT: [[EXIT]]:
; DEFAULT-NEXT: ret void
; DEFAULT-NEXT: }
;
; ----------------------------------------------------------------------------

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  reattach within %syncreg, label %latch

latch:
  %i.next = add i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, %n
  br i1 %cmp.i, label %sync, label %header, !llvm.loop !0

sync:
  sync within %syncreg, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"tapir.loop.target", i32 1}
!3 = !{!"tapir.loop.lowering.enabled"}
