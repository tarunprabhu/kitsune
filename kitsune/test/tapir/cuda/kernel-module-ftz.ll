; Check that the ftz flag can be overridden in the kernel module annotations.
; TODO: Should also add a check that this has the expected effect on the kernel
; module as well.
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s -check-prefixes=ALL,DEFAULT
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:     -cuabi-ftz \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s -check-prefixes=ALL,FTZ
;
; ALL: !llvm.module.flags = !{{{.*}}![[FTZ:[0-9]+]]{{.*}}}
;
; DEFAULT: ![[FTZ]] = !{i32 4, !"nvvm-reflect-ftz", i32 0}
; FTZ: ![[FTZ]] = !{i32 4, !"nvvm-reflect-ftz", i32 1}

define void @f1(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr i64, ptr %c, i64 %i
  store i64 %n, ptr %arrayidx, align 4
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
