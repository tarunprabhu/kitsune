; Kernel functions must have a single return instruction
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | not opt -passes='emb-finalize-reductions' 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: function must have a unique return

define void @f(ptr %res) !kit.func !0 {
  %l = alloca i64
  %v = load i64, ptr %l
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1024, i32 5, ptr %res, i64 %v, i32 10, i32 8, i64 0, ptr @sum)
  %cmpn = icmp eq i64 %v, 1
  br i1 %cmpn, label %one, label %two

one:
  ret void

two:
  ret void
}

declare void @sum(ptr, i64)

!0 = distinct !{!0, !1}
!1 = !{!"kit.func.kernel", i32 1}
