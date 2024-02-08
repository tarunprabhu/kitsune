; Check that any non-constant global variables are in the correct address space
; in the kernel module
;
; RUN: opt --tapir=hip -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @v137 = {{[A-Za-z_]+}} addrspace(1) global i32
; CHECK-DAG: @v138 = {{[A-Za-z_]+}} addrspace(1) global i32
; CHECK-DAG: @v139 = {{[A-Za-z_]+}} addrspace(1) global i32

@v137 = global i32 13, align 4
@v138 = external global i32, align 4
@v139 = internal global i32 291, align 4

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
