; Check that calls to kit.gpu.ipt are lowered correctly when the iterations per
; thread are not known at compile-time.
;
; RUN: not opt -passes=kit-lower-intrinsics-early -S %s 2>&1 | FileCheck %s
;
; CHECK: NOT YET IMPLEMENTED: Computing iterations per thread for GPU loops at runtime

@0 = internal constant ptr null

define i64 @f() {
   %ipt = call i64 @llvm.kit.gpu.ipt(i32 2, ptr @0)
   ret i64 %ipt
}
