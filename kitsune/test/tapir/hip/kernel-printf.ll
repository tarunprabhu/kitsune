; Check that calls to printf are lowered correctly.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:     --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning,emb-resolve-libdevice-calls' -S %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
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

@.str = private unnamed_addr constant [9 x i8] c"%ld: %s\0A\00", align 1
@.str.1 = private unnamed_addr constant [12 x i8] c"Hello world\00", align 1

declare i32 @printf(ptr, ...)

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %call7 = call i32 (ptr, ...) @printf(ptr @.str, i64 %i, ptr @.str.1)
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

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"tapir.loop.lowering.enabled"}
