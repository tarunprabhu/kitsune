; Check that the parameters of the outlined kernel function are in the correct
; address space. The arguments to the device functions do not need to be in any
; particular address space.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     --tapir-hip-features="+16-bit-insts" \
; RUN:     -passes='loop-spawning,emb-prepare' -S %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}} @id(
; CHECK-SAME: ptr %{{.+}})
;
; CHECK: define {{.+}} @__kithip_loop_f{{[^(]*}}(
; CHECK-SAME: ptr addrspace(1) align 1 %{{[^,]+}},
; CHECK-SAME: ptr addrspace(1) align 1 %{{[^,]+}},
; CHECK-SAME: ptr addrspace(1) align 1 %{{[^)]+}})

define ptr @id(ptr %p) {
  ret ptr %p
}

define void @f(ptr %c, ptr %a, ptr %b, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %ptra = getelementptr i32, ptr %a, i64 %i
  %0 = load i32, ptr %ptra
  %ptrb = getelementptr i32, ptr %b, i64 %i
  %1 = load i32, ptr %ptrb
  %2 = add i32 %0, %1
  %3 = inttoptr i32 %2 to ptr
  %4 = tail call ptr @id(ptr %3)
  %ptrc = getelementptr i32, ptr %c, i64 %i
  store ptr %4, ptr %ptrc, align 4
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
