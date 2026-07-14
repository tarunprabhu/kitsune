; REQUIRES: kitsune-opencilk
;
; Check that intrinsics that map to Kitsune's opencilk runtime are lowered
; correctly. If more intrinsics are created, they should be added here to test
; basic intrinsic lowering.
;
; RUN: opt --tapir=opencilk -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: call void @__kitocilk_initialize() #[[INITIALIZE:[0-9]+]]
; CHECK-NEXT: call void @__kitrt_enable_verbose_mode() #[[VERBOSE:[0-9]+]]
; CHECK-NEXT: call i64 @__kitocilk_num_workers() #[[THREADS:[0-9]+]]
; CHECK-NEXT: call i64 @__kitocilk_worker_id() #[[ID:[0-9]+]]
; CHECK-NEXT: call i64 @__kitocilk_reduce_num_partials(i64 %[[N]]) #[[PARTIALS:[0-9]+]]
; CHECK-NEXT: call void @__kitocilk_finalize() #[[FINALIZE:[0-9]+]]
;
; CHECK-DAG: attributes #[[INITIALIZE]] = { "initialize" }
; CHECK-DAG: attributes #[[VERBOSE]] = { "verbose" }
; CHECK-DAG: attributes #[[THREADS]] = { "threads" }
; CHECK-DAG: attributes #[[PARTIALS]] = { "partials" }
; CHECK-DAG: attributes #[[FINALIZE]] = { "finalize" }
; CHECK-DAG: attributes #[[ID]] = { "id" }

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]

define void @f(ptr %buf, i64 %n) {
  call void @llvm.kit.runtime.initialize(i32 8) #0
  call void @llvm.kit.runtime.set.verbose(i32 2, i8 1) #1
  call void @llvm.kit.runtime.set.verbose(i32 2, i8 0) #1
  %numThreads = call i64 @llvm.kit.cpu.num.threads(i32 8) #2
  %threadId = call i64 @llvm.kit.cpu.thread.id(i32 8) #5
  %numPartials = call i64 @llvm.kit.reduce.num.partials(i32 8, i64 %n) #3
  call void @llvm.kit.runtime.finalize(i32 8) #4
  ret void
}

attributes #0 = { "initialize" }
attributes #1 = { "verbose" }
attributes #2 = { "threads" }
attributes #3 = { "partials" }
attributes #4 = { "finalize" }
attributes #5 = { "id" }
