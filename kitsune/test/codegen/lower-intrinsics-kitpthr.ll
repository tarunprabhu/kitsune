; Check that intrinsics that map to Kitsune's phtreads runtime are lowered
; correctly. If more intrinsics are created, they should be added here to test
; basic intrinsic lowering.
;
; RUN: opt --tapir=pthreads -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]

; CHECK-LABEL: @f
; CHECK-NEXT: call void @__kitpthr_initialize()
; CHECK-NEXT: %[[CTX:[0-9]+]] = call ptr @__kitpthr_launch(ptr @f, i64 0, i64 128, i64 1, ptr @gbuf)
; CHECK-NEXT: call void @__kitpthr_sync(ptr %[[CTX]])
; CHECK-NEXT: call void @__kitpthr_finalize()

define void @f(ptr %buf, i64 %n) {
  call void @llvm.kit.initialize(i32 1024)
  %1 = call ptr @llvm.kit.async.launch.threads(i32 1024, ptr @f, i64 0, i64 128, i64 1, ptr @gbuf)
  call void @llvm.kit.sync.threads(i32 1024, ptr %1)
  call void @llvm.kit.finalize(i32 1024)
  ret void
}
