; Check that the names of global variables are fixed before generating the
; fat binary. ptxas does not allow names containing "."'s which can be present
; in code that is outlined into the kernel module, especially in templated C++
; code.
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @__kitcu__nwnm__v137_suffix = {{.*}}global i32
; CHECK-DAG: @__kitcu__nwnm__v138_const = internal constant [4 x i32]

@v137.suffix = external global i32, align 4
@v138.const = constant [4 x i32] [i32 10, i32 21, i32 42, i32 93]

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %0 = load i32, ptr @v137.suffix, align 4
  %1 = getelementptr i32, ptr @v138.const, i64 %i
  %2 = load i32, ptr %1, align 4
  %3 = add i32 %0, %2
  %arrayidx = getelementptr i32, ptr %c, i64 %i
  store i32 %3, ptr %arrayidx, align 4
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
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.depth", i32 1}
!5 = !{!"tapir.loop.perfect.level", i32 1}
