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
; CHECK-NEXT: call void @__kitqthr_launch(ptr nonnull @f, i64 0, i64 128, ptr @gbuf) #[[LAUNCH:[0-9]+]]
; CHECK-NEXT: call i64 @__kitqthr_num_workers() #[[THREADS:[0-9]+]]
; CHECK-NEXT: call i64 @__kitqthr_worker_id() #[[ID:[0-9]+]]
; CHECK-NEXT: call i64 @__kitqthr_reduce_num_partials(i64 %[[N]]) #[[PARTIALS:[0-9]+]]
; CHECK-NEXT: call void @__kitqthr_finalize() #[[FINALIZE:[0-9]+]]
; CHECK-NEXT: call void @__kitqthr_initialize() #[[INITIALIZE:[0-9]+]]

; CHECK-DAG: #[[FINALIZE]] = { "finalize" }
; CHECK-DAG: #[[ID]] = { "id" }
; CHECK-DAG: #[[INITIALIZE]] = { "initialize" }
; CHECK-DAG: #[[LAUNCH]] = { "launch" }
; CHECK-DAG: #[[PARTIALS]] = { "partials" }
; CHECK-DAG: #[[THREADS]] = { "threads" }

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]

define void @f(ptr %buf, i64 %n) {
  call void @llvm.kit.cpu.threads.launch(i32 32, ptr nonnull @f, i64 0, i64 128, ptr @gbuf) #0
  %numThreads = call i64 @llvm.kit.cpu.num.threads(i32 32) #1
  %threadID = call i64 @llvm.kit.cpu.thread.id(i32 32) #2
  %numPartials = call i64 @llvm.kit.reduce.num.partials(i32 32, i64 %n) #3
  call void @llvm.kit.runtime.finalize(i32 32) #4
  call void @llvm.kit.runtime.initialize(i32 32) #5
  ret void
}

attributes #0 = { "launch" }
attributes #1 = { "threads" }
attributes #2 = { "id" }
attributes #3 = { "partials" }
attributes #4 = { "finalize" }
attributes #5 = { "initialize" }
