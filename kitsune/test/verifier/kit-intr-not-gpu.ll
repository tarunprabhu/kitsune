; Certain Kitsune intrinsics may only be called with a GPU-centric tapir target.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK-COUNT-4: TTID argument in call is not GPU-centric

define void @f(ptr %stream) {
  call void @llvm.kit.gpu.stream.sync(i32 0, ptr %stream)
  call void @llvm.kit.gpu.stream.sync(i32 1, ptr %stream)
  call void @llvm.kit.gpu.stream.sync(i32 32, ptr %stream)
  call void @llvm.kit.gpu.stream.sync(i32 2048, ptr %stream)
  ret void
}
