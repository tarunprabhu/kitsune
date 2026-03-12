; Non-constant global variabes used in a kernel (or device) function must be
; in a specific address space. However, simply setting the address space is not
; sufficient since we need to ensure that the change in address space does not
; affect the instructions in the body. For instance, if the global is passed to
; a device function, there may be a type mismatch if the device function does
; not expect pointers in a specific address space. In order to deal with this,
; we cast away the address space in every use of the global.
;
; RUN: opt --tapir=hip -passes='loop-spawning,emb-prepare' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: @[[GV:.+]] = {{.*}}addrspace(1) global [16 x i64]
;
; CHECK-LABEL: define {{.+}} @id(
; CHECK-SAME: ptr %[[ARG0:[^,]+]],
; CHECK-SAME: i64 %[[ARG1:[^)]+]])
; CHECK: getelementptr i64, ptr addrspacecast (ptr addrspace(1) @[[GV]] to ptr), i64 %[[ARG1]]
;
; CHECK-LABEL: define {{.+}} @__kithip_loop_f
; CHECK: %[[IV:.+]] = phi i64
; CHECK: %[[V0:.+]] = tail call fastcc i64 @id(ptr addrspacecast (ptr addrspace(1) @[[GV]] to ptr), i64 %[[IV]])
; CHECK: %[[V1:.+]] = ptrtoint ptr addrspacecast (ptr addrspace(1) @[[GV]] to ptr) to i64
; CHECK: %[[V2:.+]] = getelementptr i64, ptr addrspacecast (ptr addrspace(1) @[[GV]] to ptr), i64 %[[IV]]
; CHECK: %[[V3:.+]] = load i64, ptr addrspacecast (ptr addrspace(1) @[[GV]] to ptr)
; CHECK: ptrtoint ptr %[[V2]] to i64

@v137 = external global [16 x i64]

define i64 @id(ptr %p, i64 %iv) {
entry:
  %0 = getelementptr i64, ptr @v137, i64 %iv
  %1 = load i64, ptr %0
  ret i64 %1
}

define void @f(ptr %a, ptr %b, ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %0 = tail call i64 @id(ptr @v137, i64 %i)
  %1 = ptrtoint ptr @v137 to i64
  %2 = getelementptr i64, ptr @v137, i64 %i
  %3 = load i64, ptr @v137
  %4 = add i64 %0, %1
  %5 = ptrtoint ptr %2 to i64
  %6 = mul i64 %4, %5
  %7 = sub i64 %6, %3
  store i64 %7, ptr @v137
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
