; Check that alloca instructions have the correct address space annotation
; added to them.
;
; RUN: opt --tapir=hip -passes='loop-spawning,emb-prepare' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: %[[A1:.+]] = alloca i64, align 8, addrspace(5)
; CHECK: %[[A2:.+]] = alloca i64, align 8, addrspace(5)
; CHECK: %[[CST1:.+]] = addrspacecast ptr addrspace(5) %[[A1]] to ptr
; CHECK: %[[CST2:.+]] = addrspacecast ptr addrspace(5) %[[A2]] to ptr
; CHECK: %[[V0:.+]] = ptrtoint ptr %[[CST1]] to i64
; CHECK: %[[V1:.+]] = ptrtoint ptr %[[CST2]] to i64

%"class.kitsune::mobile_ptr" = type { ptr addrspace(67) }

@stash = global ptr zeroinitializer

define void @_ZL20initialize_variablesiN7kitsune10mobile_ptrIfEES1_(i64 %nelr, ptr %variables, ptr %ff_variable) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %invariant.gep = getelementptr float, ptr %variables, i64 %nelr
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %a1 = alloca i64, align 8
  %a2 = alloca i64, align 8
  %0 = ptrtoint ptr %a1 to i64
  %1 = ptrtoint ptr %a2 to i64
  %2 = add i64 %0, %1
  %3 = add i64 %2, %i
  %4 = load float, ptr %ff_variable, align 4
  %arrayidx.i = getelementptr float, ptr %variables, i64 %3
  store float %4, ptr %arrayidx.i, align 4
  reattach within %syncreg, label %latch

latch:
  %i.next = add i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, %nelr
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
