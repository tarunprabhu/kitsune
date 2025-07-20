; Check that the kit-embs pass does not create any embedded globals when run
; with a tapir target that does not generate bitcode.
;
; RUN: opt --tapir=serial -passes=kit-embs,verify -S -o - %s \
; RUN:     | FileCheck %s
;
; CHECK-NOT: kit_bc
; CHECK-NOT: kit_fb
; CHECK-NOT: kit_tt

@gd = global i32 120
@gh = global i32 240

define void @d0() #0 {
  ret void
}

define void @d1() #1 {
  call void @d2()
  ret void
}

define void @d2() {
  call void @d3(ptr @gd)
  ret void
}

define void @d3(ptr %0) {
  ret void
}

define void @h4() {
  ret void
}

define void @d5(i64 %i) {
  ret void
}

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i64 %n, 0
  br i1 %cmp5, label %preheader, label %forall.sync

preheader:
  br label %forall.detach

forall.detach:
  %indvars.iv = phi i64 [ 0, %preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  call void @d5(i64 %indvars.iv)
  reattach within %syncreg, label %forall.inc

forall.inc:
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}

attributes #0 = { kit_device }
attributes #1 = { kit_device }

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}
