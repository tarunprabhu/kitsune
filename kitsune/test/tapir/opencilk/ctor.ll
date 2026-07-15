; Check that the global ctor calls the appropriate functions in Kitsune's
; runtime depending on the command line options passed.
;
; ------------------------------------------------------------------------------
; This runs the minimum number of passes that must be run to exercise the ctor
; generation code.
;
; RUN: opt -passes='loop-spawning,tapir2target,kit-ctors' -S %s \
; RUN:     --tapir=opencilk \
; RUN:     --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
; RUN:     | FileCheck %s -check-prefix DEFAULT
;
; ------------------------------------------------------------------------------
; This runs the standard sequence that will be run during most compilations.
; For the ctor to be inserted, we need to check if elements of OpenCilk's
; runtime are used in the transformed code. The detector checks for the use of
; several different elements since LLVM's optimization passes may obfuscate some
; uses. This runs the full lowering pipeline that will have run several
; optimizations. It is not guaranteed to exercise all paths in the detector, but
; it is, arguably, better than nothing.
;
; RUN: opt -passes='kit-lowering<O3>' -S %s \
; RUN:     --tapir=opencilk \
; RUN:     --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
; RUN:     | FileCheck %s -check-prefix DEFAULT
;
; ------------------------------------------------------------------------------
; DEFAULT-LABEL: @llvm.global_ctors = appending global
; DEFAULT-SAME: { i32 65535, ptr @[[CTOR:.+]], ptr null }
;
; DEFAULT-LABEL: @llvm.global_dtors = appending global
; DEFAULT-SAME: { i32 65535, ptr @[[DTOR:.+]], ptr null }
;
; DEFAULT: define internal void @[[CTOR]]()
; DEFAULT-NEXT: [[ENTRY:.+]]:
; DEFAULT-NEXT: call {{.+}} @llvm.kit.runtime.initialize(i32 8)
; DEFAULT-NEXT: br label %[[EXIT:.+]]
; DEFAULT-EMPTY:
; DEFAULT-NEXT: [[EXIT]]:
; DEFAULT-NEXT: ret void
; DEFAULT-NEXT: }
;
; DEFAULT: define internal void @[[DTOR]]()
; DEFAULT-NEXT: [[ENTRY:.+]]:
; DEFAULT-NEXT: call {{.+}} @llvm.kit.runtime.finalize(i32 8)
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
  %addr = getelementptr i64, ptr %c, i64 %i
  store i64 %i, ptr %addr
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
!1 = !{!"tapir.loop.spawn.strategy", i32 2}
!2 = !{!"tapir.loop.target", i32 8}
!3 = !{!"tapir.loop.lowering.enabled"}
