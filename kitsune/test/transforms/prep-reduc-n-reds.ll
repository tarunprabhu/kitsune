; When more than one reduction is performed in a tapir reduction loop, check
; that:
;
;   - Distinct local reduction buffers are allocated (and freed) for each
;
;   - This final reduction is performed in the same order as in the original
;     code (this isn't strictly necessary, but we generate code in this order,
;     so we may as well check for it)
;
; NOTE: The loop here has the pthreads tapir target set on it. This is only
; because we need to set a non-serial tapir target on the loop. We need to test
; the GPU tapir targets separately because the transformations needed for those
; will be different.
;
; RUN: opt -passes=kit-prepare -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N:[^)]+]]
;
; CHECK: %[[R1:.+]] = alloca i64
; CHECK: %[[R2:.+]] = alloca i64
; CHECK: %[[R3:.+]] = alloca i32
; CHECK: %[[R4:.+]] = alloca i32
;
; CHECK: %[[NREDS:.+]] = call i64 @llvm.kit.cpu.num.threads(i32 1024)
;
; CHECK: %[[IV_O:.+]] = phi i64
; CHECK-NEXT: detach within
;
; CHECK: %[[LOCAL1:.+]] = tail call ptr @malloc(i64 8)
; CHECK: %[[LOCAL2:.+]] = tail call ptr @malloc(i64 8)
; CHECK: %[[LOCAL3:.+]] = tail call ptr @malloc(i64 4)
; CHECK: %[[LOCAL4:.+]] = tail call ptr @malloc(i64 4)
;
; CHECK: %[[IV_I:.+]] = phi i64
; CHECK: %[[J32:.+]] = trunc i64 %[[IV_I]] to i32
;
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0
; CHECK-SAME: i32 1024, i32 5, ptr %[[LOCAL1]], i32 8, i64 %[[IV_I]], i64 0, ptr @sum.i64
;
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0
; CHECK-SAME: i32 1024, i32 1, ptr %[[LOCAL2]], i32 8, i64 %[[IV_I]], i64 1, ptr @and.i64
;
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0
; CHECK-SAME: i32 1024, i32 1, ptr %[[LOCAL3]], i32 4, i32 %[[J32]], i32 1, ptr @and.i32
;
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce.0
; CHECK-SAME: i32 1024, i32 5, ptr %[[LOCAL4]], i32 4, i32 %[[J32]], i32 0, ptr @sum.i32
;
; CHECK: %[[PARTIAL1:.+]] = load i64, ptr %[[LOCAL1]]
; CHECK-NEXT: atomicrmw add ptr %[[R1]], i64 %[[PARTIAL1]] monotonic
; CHECK-NEXT: tail call void @free(ptr %[[LOCAL1]])
;
; CHECK-NEXT: %[[PARTIAL2:.+]] = load i64, ptr %[[LOCAL2]]
; CHECK-NEXT: atomicrmw and ptr %[[R2]], i64 %[[PARTIAL2]] monotonic
; CHECK-NEXT: tail call void @free(ptr %[[LOCAL2]])
;
; CHECK-NEXT: %[[PARTIAL3:.+]] = load i32, ptr %[[LOCAL3]]
; CHECK-NEXT: atomicrmw and ptr %[[R3]], i32 %[[PARTIAL3]] monotonic
; CHECK-NEXT: tail call void @free(ptr %[[LOCAL3]])
;
; CHECK-NEXT: %[[PARTIAL4:.+]] = load i32, ptr %[[LOCAL4]]
; CHECK-NEXT: atomicrmw add ptr %[[R4]], i32 %[[PARTIAL4]] monotonic
; CHECK-NEXT: tail call void @free(ptr %[[LOCAL4]])

declare void @sum.i64(ptr %res, i64 %v)

declare void @and.i64(ptr %res, i64 %v)

declare void @sum.i32(ptr %res, i32 %v)

declare void @and.i32(ptr %res, i32 %v)

define void @f(i64 %n) {
entry:
  %r1 = alloca i64
  %r2 = alloca i64
  %r3 = alloca i32
  %r4 = alloca i32
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %entry ], [ %inc.j, %for.j.latch ]
  detach within %syncreg, label %for.j.body, label %for.j.latch

for.j.body:
  %j32 = trunc i64 %j to i32
  call void(i32, i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 1024, i32 5, ptr %r1, i32 8, i64 %j, i64 0, ptr @sum.i64)
  call void(i32, i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 1024, i32 1, ptr %r2, i32 8, i64 %j, i64 1, ptr @and.i64)
  call void(i32, i32, ptr, i32, i32, i32, ptr, ...) @llvm.kit.reduce.0(i32 1024, i32 1, ptr %r3, i32 4, i32 %j32, i32 1, ptr @and.i32)
  call void(i32, i32, ptr, i32, i32, i32, ptr, ...) @llvm.kit.reduce.0(i32 1024, i32 5, ptr %r4, i32 4, i32 %j32, i32 0, ptr @sum.i32)
  reattach within %syncreg, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

for.j.exit:
  sync within %syncreg, label %for.j.end

for.j.end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1024}
!1 = !{!"tapir.loop.reduction"}
!2 = distinct !{!2, !0, !1}
