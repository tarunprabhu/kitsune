; Certain Kitsune intrinsics may only be called with a CPU-centric tapir target.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK-COUNT-4: TTID argument in call is not CPU-centric

define void @f() {
  %1 = call i64 @llvm.kit.cpu.num.threads(i32 0)
  %2 = call i64 @llvm.kit.cpu.num.threads(i32 1)
  %3 = call i64 @llvm.kit.cpu.num.threads(i32 2)
  %4 = call i64 @llvm.kit.cpu.num.threads(i32 2048)
  ret void
}
