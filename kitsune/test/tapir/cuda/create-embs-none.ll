; Check that the kit-embs pass does not create any embedded globals when there
; are no device functions, nor tapir loops in the module.
;
; RUN: opt --tapir=cuda -passes=kit-embs,verify -S -o - %s \
; RUN:     | FileCheck %s
;
; CHECK-NOT: kit_bc
; CHECK-NOT: kit_fb
; CHECK-NOT: kit_tt

@gd = global i32 120
@gh = global i32 240

define void @d1(i64 %i) {
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
  br label %forall.body

forall.body:
  call void @d1(i64 %indvars.iv)
  br label %forall.inc

forall.inc:
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
