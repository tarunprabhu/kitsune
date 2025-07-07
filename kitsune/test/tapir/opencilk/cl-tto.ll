; Check that the command line options make it to the options objects
;
; RUN: opt --tapir=opencilk -passes="tapir-lowering<O2>" -o /dev/null %s \
; RUN:     --tapir-verbose \
; RUN:     --kitrt-verbose \
; RUN:     --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc 2>&1 \
; RUN:     | FileCheck %s -check-prefixes ALL,CHECK
;
; ALL: 'opencilk' tapir target options
; CHECK:     Runtime verbose: 1
; CHECK:     Opencilk bitcode file: {{.+}}/libopencilk-abi.bc

target triple = "x86_64-pc-linux-gnu"

define void @f(ptr %c, i32 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %preheader, label %forall.sync

preheader:
  %wide.trip.count = zext nneg i32 %n to i64
  br label %forall.detach

forall.detach:
  %indvars.iv = phi i64 [ 0, %preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store i32 %n, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}
!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}
