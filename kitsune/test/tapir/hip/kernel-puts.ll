; Check that calls to puts are lowered correctly.
;
; RUN: opt --tapir=hip --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning,emb-resolve-libdevice-calls' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; Calls to puts(s) are lowered to the equivalent of printf("%s\n", s). This will
; result in a global variable being created for the format string. The AMDGPU
; printf call construction does not seem to use that global directly, so we
; don't check for its use, just its creation.
;
; CHECK: private unnamed_addr addrspace(1) constant [4 x i8] c"%s\0A\00"
;
; CHECK: %[[IV:.+]] = phi i64
; CHECK: %[[PRINTF_ALLOC_FN:.+]] = call ptr addrspace(1) @__printf_alloc
; CHECK: %[[COND:.+]] = icmp ne ptr addrspace(1) %[[PRINTF_ALLOC_FN]], null
; CHECK: br i1 %[[COND]], label %[[BBARGS:.+]], label %[[BBEND:.+]]
; CHECK: [[BBBODY:.+]]:
; CHECK: br label %[[BBINC:.+]]
; CHECK: [[BBINC]]:
; CHECK: [[BBEND]]:
; CHECK: br label %[[BBBODY]]
; CHECK: [[BBARGS]]:
; CHECK: store i32 {{[0-9]+}}, ptr addrspace(1) %[[PRINTF_ALLOC_FN]]
; CHECK: br label %[[BBEND]]

@.str.1 = private unnamed_addr constant [7 x i8] c"Line 1\00", align 1

declare i32 @puts(ptr)

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %call1 = call i32 @puts(ptr @.str.1)
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
