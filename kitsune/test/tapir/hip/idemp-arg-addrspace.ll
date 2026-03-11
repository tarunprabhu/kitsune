; Check that running the prepare embedded bitcode pass on a kernel module that
; it has already been run on does not cause any appreciable changes.
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     --tapir-hip-features="+16-bit-insts" \
; RUN:     -passes='loop-spawning,emb-prepare,emb-prepare' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}} @id(
; CHECK-SAME: ptr %{{.+}})
;
; CHECK: define {{.+}} @__kithip_loop_f{{[^(]*}}(
; CHECK-SAME: ptr addrspace(1) align 1 %[[A:[^,]+]],
; CHECK-SAME: ptr addrspace(1) align 1 %[[B:[^,]+]],
; CHECK-SAME: ptr addrspace(1) align 1 %[[C:[^)]+]])
; CHECK: %[[CSTA:.+]] = addrspacecast ptr addrspace(1) %[[A]] to ptr
; CHECK: %[[CSTB:.+]] = addrspacecast ptr addrspace(1) %[[B]] to ptr
; CHECK: %[[CSTC:.+]] = addrspacecast ptr addrspace(1) %[[C]] to ptr
; CHECK: %[[IV:.+]] = phi i64
; CHECK: getelementptr {{.+}}, ptr %[[CSTA]], i64 %[[IV]]
; CHECK: getelementptr {{.+}}, ptr %[[CSTB]], i64 %[[IV]]
; CHECK: getelementptr {{.+}}, ptr %[[CSTC]], i64 %[[IV]]

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
