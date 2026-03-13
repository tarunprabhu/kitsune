; Check that any constant global variables are copied into the kernel module
; but with the linkage set to internal, regardless of what the linkage is in
; the host module
;
; RUN: opt --tapir=hip -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @v137 = internal {{.*}}constant i32 921
; CHECK-DAG: @v138 = internal {{.*}}constant i32 11
; CHECK-DAG: @v139 = internal {{.*}}constant i32 46

@v137 = constant i32 921, align 4
@v138 = private constant i32 11, align 4
@v139 = internal constant i32 46, align 4

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %0 = load i32, ptr @v137, align 4
  %1 = load i32, ptr @v138, align 4
  %2 = add i32 %0, %1
  %3 = load i32, ptr @v139, align 4
  %4 = add i32 %2, %3
  %arrayidx = getelementptr i32, ptr %c, i64 %i
  store i32 %4, ptr %arrayidx, align 4
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

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}
