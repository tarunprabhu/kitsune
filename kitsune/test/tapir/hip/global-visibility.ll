; Check that non-constant global variables used in a tapir loop are cloned into
; the kernel module with the correct visibility.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @v137 = protected {{.*}}global i32

@v137 = external local_unnamed_addr global i32, align 4

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %0 = load i32, ptr @v137, align 4
  %arrayidx = getelementptr i32, ptr %c, i64 %i
  store i32 %0, ptr %arrayidx, align 4
  reattach within %syncreg, label %latch

latch:
  %i.next = add nuw i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, %n
  br i1 %cmp.i, label %sync, label %header, !llvm.loop !0

sync:
  sync within %syncreg, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}
