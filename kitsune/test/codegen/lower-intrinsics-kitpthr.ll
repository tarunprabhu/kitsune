; Check that intrinsics that map to Kitsune's pthreads runtime are lowered
; correctly. If more intrinsics are created, they should be added here to test
; basic intrinsic lowering.
;
; RUN: opt --tapir=pthreads -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]

; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: call void @__kitpthr_initialize()
; CHECK-NEXT: call void @__kitrt_enable_verbose_mode()
; CHECK-NEXT: %[[CTX:[0-9]+]] = call ptr @__kitpthr_launch(ptr @f, i64 0, i64 128, i64 1, ptr @gbuf)
; CHECK-NEXT: call void @__kitpthr_sync(ptr %[[CTX]])
; CHECK-NEXT: call i32 @__kitpthr_num_threads()
; CHECK-NEXT: call i64 @__kitpthr_reduce_num_partials(i64 %[[N]])
; CHECK-NEXT: call void @__kitpthr_finalize()

define void @f(ptr %buf, i64 %n) {
  call void @llvm.kit.runtime.initialize(i32 1024)
  call void @llvm.kit.runtime.set.verbose(i32 2, i8 1)
  call void @llvm.kit.runtime.set.verbose(i32 2, i8 0)
  %1 = call ptr @llvm.kit.async.cpu.threads.launch(i32 1024, ptr @f, i64 0, i64 128, i64 1, ptr @gbuf)
  call void @llvm.kit.cpu.threads.sync(i32 1024, ptr %1)
  %numThreads = call i32 @llvm.kit.cpu.num.threads(i32 1024)
  %numPartials = call i64 @llvm.kit.reduce.num.partials(i32 1024, i64 %n)
  call void @llvm.kit.runtime.finalize(i32 1024)
  ret void
}
