; Check that intrinsics that map to Kitsune's pthreads runtime are lowered
; correctly. If more intrinsics are created, they should be added here to test
; basic intrinsic lowering.
;
; We also check that any attributes attached to the arguments and the call
; itself are also attached to the lowered call.
;
; RUN: opt --tapir=pthreads -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: %[[BUNDLE:.+]] = alloca { ptr, float }
; CHECK-NEXT: %[[OFF0:.+]] = getelementptr inbounds { ptr, float }, ptr %[[BUNDLE]], i32 0, i32 0
; CHECK-NEXT: store ptr null, ptr %[[OFF0]]
; CHECK-NEXT: %[[OFF1:.+]] = getelementptr inbounds { ptr, float }, ptr %[[BUNDLE]], i32 0, i32 1
; CHECK-NEXT: store float 1.500000e+00, ptr %[[OFF1]]
; CHECK-NEXT: %[[CTX:.+]] = call ptr @__kitpthr_async_launch(ptr nonnull @f, i64 0, i64 128, ptr %[[BUNDLE]], i32 16) #[[LAUNCH:[0-9]+]]
; CHECK-NEXT: call i64 @__kitpthr_num_threads() #[[THREADS:[0-9]+]]
; CHECK-NEXT: call i64 @__kitpthr_thread_id() #[[ID:[0-9]+]]
; CHECK-NEXT: call void @__kitpthr_sync(ptr nonnull %[[CTX]]) #[[SYNC:[0-9]+]]
;
; CHECK-DAG: #[[ID]] = { "id" }
; CHECK-DAG: #[[LAUNCH]] = { "launch" }
; CHECK-DAG: #[[SYNC]] = { "sync" }
; CHECK-DAG: #[[THREADS]] = { "threads" }

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]

define void @f(ptr %buf, i64 %n) {
  %ctx = call ptr(i32, ptr, i64, i64, ...) @llvm.kit.async.cpu.threads.launch(i32 1024, ptr nonnull @f, i64 0, i64 128, ptr null, float 1.5) #0
  %numThreads = call i64 @llvm.kit.cpu.num.threads(i32 1024) #1
  %threadID = call i64 @llvm.kit.cpu.thread.id(i32 1024) #2
  call void @llvm.kit.cpu.threads.sync(i32 1024, ptr nonnull %ctx) #3
  ret void
}

attributes #0 = { "launch" }
attributes #1 = { "threads" }
attributes #2 = { "id" }
attributes #3 = { "sync" }
