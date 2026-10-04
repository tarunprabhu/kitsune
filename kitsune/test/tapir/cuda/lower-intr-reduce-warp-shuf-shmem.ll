; Check that the kit.gpu.reduce intrinsic with the
; `GPUWarpShuffleWithSharedMemory` strategy is lowered correctly.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | not --crash opt -passes=emb-lower-intrinsics 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: NOT YET IMPLEMENTED: ReduceWarpShuffleWithSharedMemory

declare void @sum(ptr, i32)

define void @f(ptr %result, i64 %v) {
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.gpu.reduce(i32 2, i32 1, ptr %result, i64 %v, i32 6, i32 8, i64 0, ptr @sum)
  ret void
}
