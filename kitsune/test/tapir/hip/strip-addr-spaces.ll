; Check that Kitsune's address spaces are stripped from the kernel module when
; the prepare pass is run.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     --passes='loop-spawning,emb-prepare' -S %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-NOT: addrspace(67)

define void @vecadd(ptr addrspace(67) %c, ptr addrspace(67) %a, ptr addrspace(67) %b, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx.a = getelementptr double, ptr addrspace(67) %a, i64 %i
  %0 = load double, ptr addrspace(67) %arrayidx.a, align 8
  %arrayidx.b = getelementptr double, ptr addrspace(67) %b, i64 %i
  %1 = load double, ptr addrspace(67) %arrayidx.b, align 8
  %add = fadd double %0, %1
  %arrayidx.c = getelementptr double, ptr addrspace(67) %c, i64 %i
  store double %add, ptr addrspace(67) %arrayidx.c, align 8
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
