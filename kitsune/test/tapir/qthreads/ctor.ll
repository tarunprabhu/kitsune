; Check that the global ctor calls the appropriate functions in Kitsune's
; runtime depending on the command line arguments passed.
;
; RUN: opt --tapir=qthreads -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     | FileCheck %s -check-prefix DEFAULT
;
; DEFAULT: @llvm.global_ctors = appending global
; DEFAULT-SAME: { i32 65536, ptr @[[CTOR:[.]kitqthr[.]ctor.*]], ptr null }
;
; DEFAULT: define {{.*}} @[[DTOR:[.]kitqthr[.]dtor.*]]{{[ ]*}}(
; DEFAULT: call {{.+}} @llvm.kit.runtime.finalize(i32 32)
;
; DEFAULT: define {{.+}} @[[CTOR]]
; DEFAULT-NEXT: [[ENTRY:.+]]:
; DEFAULT-NEXT: call {{.+}} @llvm.kit.runtime.initialize(i32 32)
; DEFAULT-NEXT: call {{.+}} @llvm.kit.runtime.set.verbose(i32 32, i8 0)
; DEFAULT-NEXT: call {{.+}}atexit(ptr @[[DTOR]])
; DEFAULT-NEXT: ret void
; DEFAULT-NEXT: }
;
; RUN: opt --tapir=qthreads -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     --tapir-verbose \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; RUN: opt --tapir=qthreads -passes='loop-spawning,kit-ctors' -S %s \
; RUN:     --kitrt-verbose \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; VERBOSE-LABEL: define {{.+}} @.kitqthr.ctor
; VERBOSE: call {{.+}} @llvm.kit.runtime.set.verbose(i32 32, i8 1)
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
!1 = !{!"tapir.loop.spawn.strategy", i32 4}
!2 = !{!"tapir.loop.target", i32 32}
!3 = !{!"tapir.loop.lowering.enabled"}
