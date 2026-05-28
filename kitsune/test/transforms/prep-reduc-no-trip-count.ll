; Tapir reduction loops must have a computable finite trip count.
;
; NOTE: In assert builds, an assertion failure will occur. We don't bother
; checking for the specific message since that is not important. For non-assert
; builds, we check for the actual error since that is what a user will see.
;
; RUN: %if asserts %{ \
; RUN:   not --crash opt -passes='kit-reductions' -S %s 2>&1 \
; RUN:       | FileCheck %s --check-prefix ASSERT \
; RUN: %} %else %{ \
; RUN:   not opt -passes='kit-reductions' -S %s 2>&1 \
; RUN:       | FileCheck %s --check-prefix ERRORs \
; RUN: %}
;
; ASSERT: Assertion {{.+}} failed
; ERROR: tapir reduction loop header has its address taken

declare void @sum(ptr %res, i64 %v)

define void @acc(ptr %ptrn) {
entry:
  %result = alloca i64
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %entry ], [ %inc.j, %for.j.latch ]
  detach within %syncreg, label %for.j.body, label %for.j.latch

for.j.body:
  call void(i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 1024, ptr %result, i32 8, i64 %j, i64 0, ptr @sum)
  reattach within %syncreg, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %n = load i64, ptr %ptrn
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
