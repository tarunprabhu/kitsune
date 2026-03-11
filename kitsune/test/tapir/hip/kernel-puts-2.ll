; When multiple calls to puts exist, only a single global containing the format
; string should be created.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:     --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:     -passes='loop-spawning,emb-resolve-libdevice-calls' -S %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: private unnamed_addr addrspace(1) constant [4 x i8] c"%s\0A\00"
; CHECK-NOT: private unnamed_addr addrspace(1) constant [4 x i8] c"%s\0A\00"
; CHECK: call ptr addrspace(1) @__printf_alloc(
; CHECK: call ptr addrspace(1) @__printf_alloc(

@.str.1 = private unnamed_addr constant [7 x i8] c"Line 1\00", align 1
@.str.2 = private unnamed_addr constant [7 x i8] c"Line 2\00", align 1

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
  %call2 = call i32 @puts(ptr @.str.2)
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
