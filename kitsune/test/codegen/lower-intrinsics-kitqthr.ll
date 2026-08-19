; REQUIRES: kitsune-qthreads
;
; Check that intrinsics that map to Kitsune's qthreads runtime are lowered
; correctly. If more intrinsics are created, they should be added here to test
; basic intrinsic lowering.
;
; We also check that any attributes attached to the arguments and the call
; itself are also attached to the lowered call.
;
; RUN: opt --tapir=qthreads -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: %[[BUNDLE:.+]] = alloca { ptr, float }
; CHECK-NEXT: call i64 @__kitqthr_num_workers() #[[THREADS:[0-9]+]]
; CHECK-NEXT: call i64 @__kitqthr_worker_id() #[[ID:[0-9]+]]
; CHECK-NEXT: %[[OFF0:.+]] = getelementptr inbounds { ptr, float }, ptr %[[BUNDLE]], i32 0, i32 0
; CHECK-NEXT: store ptr null, ptr %[[OFF0]]
; CHECK-NEXT: %[[OFF1:.+]] = getelementptr inbounds { ptr, float }, ptr %[[BUNDLE]], i32 0, i32 1
; CHECK-NEXT: store float 1.500000e+00, ptr %[[OFF1]]
; CHECK-NEXT: call void @__kitqthr_launch(ptr nonnull @f, i64 0, i64 128, ptr %[[BUNDLE]], i32 16) #[[LAUNCH:[0-9]+]]
; CHECK-NEXT: %[[MALLOCED:.+]] = call noalias ptr @__kitrt_malloc(i64 63) #[[MALLOC:[0-9]+]]
; CHECK-NEXT: call void @__kitrt_free(ptr %[[MALLOCED]]) #[[FREE:[0-9]+]]

; CHECK-DAG: attributes #[[ID]] = { "id" }
; CHECK-DAG: attributes #[[LAUNCH]] = { "launch" }
; CHECK-DAG: attributes #[[THREADS]] = { "threads" }
; CHECK-DAG: attributes #[[MALLOC]] = { "malloc" }
; CHECK-DAG: attributes #[[FREE]] = { "free" }

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

define void @f(ptr %buf, i64 %n) {
  %numThreads = call i64 @llvm.kit.cpu.num.threads(i32 32) #0
  %threadID = call i64 @llvm.kit.cpu.thread.id(i32 32) #1
  call void(i32, ptr, i64, i64, ...) @llvm.kit.cpu.threads.launch(i32 32, ptr nonnull @f, i64 0, i64 128, ptr null, float 1.5) #2
  %malloced = call noalias ptr @llvm.kit.cpu.malloc(i32 32, i64 63) #3
  call void @llvm.kit.cpu.free(i32 32, ptr %malloced) #4
  ret void
}

attributes #0 = { "threads" }
attributes #1 = { "id" }
attributes #2 = { "launch" }
attributes #3 = { "malloc" }
attributes #4 = { "free" }
