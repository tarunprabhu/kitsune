; Check that if the same constant global is used in two separate tapir loops,
; only a single instance of the global is created in the kernel module.
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @v137{{[^ ]*}} = internal constant [4 x i32]
; CHECK-DAG: @v138{{[^ ]*}} = internal constant [4 x i32]
; CHECK-NOT: @v137{{.*}} =
; CHECK-NOT: @v138{{.*}} =

@v137 = internal constant [4 x i32] [i32 991, i32 0, i32 11, i32 97], align 4
@v138 = private constant [4 x i32] [i32 13, i32 17, i32 91, i32 23], align 4

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %0 = getelementptr i32, ptr @v137, i64 %i
  %1 = load i32, ptr %0, align 4
  %2 = getelementptr i32, ptr @v138, i64 %i
  %3 = load i32, ptr %2, align 4
  %4 = add i32 %1, %3
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

define void @g(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %0 = getelementptr i32, ptr @v137, i64 %i
  %1 = load i32, ptr %0, align 4
  %2 = getelementptr i32, ptr @v138, i64 %i
  %3 = load i32, ptr %2, align 4
  %4 = sub i32 %1, %3
  %arrayidx = getelementptr i32, ptr %c, i64 %i
  store i32 %4, ptr %arrayidx, align 4
  reattach within %syncreg, label %latch

latch:
  %i.next = add i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, %n
  br i1 %cmp.i, label %sync, label %header, !llvm.loop !4

sync:
  sync within %syncreg, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = distinct !{!4, !1, !2, !3}
