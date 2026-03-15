; Check that the tapir target adds the expected Kitsune-specific module-level
; metadata to the kernel module.
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; The module identifier is generated a specific way. We don't really need it to
; be exactly what it is, but might as well check it.
;
; CHECK: ModuleID = '__kitnv_kernel-module-kit-md.ll'
;
; CHECK: target triple = "nvptx64-nvidia-cuda"
;
; CHECK: define {{.*}}@[[F1:__kitcu_loop_f1[^(]*]](
; CHECK: define {{.*}}@[[F2:__kitcu_loop_f2[^(]*]](
;
; CHECK: !kit.module.device.module.flags = !{![[MDTT:[0-9]+]], ![[MDNAME:[0-9]+]]}
; CHECK: !llvm.module.flags = !{{{.*}}![[FTZ:[0-9]+]]{{.*}}}
;
; CHECK-DAG: ![[FTZ]] = !{i32 4, !"nvvm-reflect-ftz", i32 0}
; CHECK-DAG: ![[MDTT]] = !{i32 2}
; CHECK-DAG: ![[MDNAME]] = !{!"__kitnv_kernel-module-kit-md.ll"}

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

define void @f2(ptr %c, i64 %n) {
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
  br i1 %cmp.i, label %sync, label %header, !llvm.loop !4

sync:
  sync within %syncreg, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !5, !6}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = distinct !{!4, !1, !2, !3, !5, !6}
!5 = !{!"tapir.loop.perfect.depth", i32 1}
!6 = !{!"tapir.loop.perfect.level", i32 1}
