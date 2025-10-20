; Check that Kitsune's address spaces are stripped from the kernel module when
; the prepare pass is run.
;
; RUN: opt --tapir=hip --passes='tapir-lowering<O2>,emb-prepare' -S %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-NOT: addrspace(67)

target triple = "x86_64-unknown-linux-gnu"

define void @vecadd(ptr addrspace(67) %c, ptr addrspace(67) %a, ptr addrspace(67) %b, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %forall.detach

forall.detach:
  %i = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %arrayidx.a = getelementptr double, ptr addrspace(67) %a, i64 %i
  %0 = load double, ptr addrspace(67) %arrayidx.a, align 8
  %arrayidx.b = getelementptr double, ptr addrspace(67) %b, i64 %i
  %1 = load double, ptr addrspace(67) %arrayidx.b, align 8
  %add = fadd double %0, %1
  %arrayidx.c = getelementptr double, ptr addrspace(67) %c, i64 %i
  store double %add, ptr addrspace(67) %arrayidx.c, align 8
  reattach within %syncreg, label %forall.inc

forall.inc:
  %inc = add nuw i64 %i, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"tapir.loop.target", i32 4}
