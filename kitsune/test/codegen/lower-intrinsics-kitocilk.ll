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
; CHECK-NEXT: call i64 @__kitocilk_num_workers() #[[THREADS:[0-9]+]]
; CHECK-NEXT: call i64 @__kitocilk_worker_id() #[[ID:[0-9]+]]
; CHECK-NEXT: call void @__kitocilk_finalize() #[[FINALIZE:[0-9]+]]
; CHECK-NEXT: call void @__kitocilk_initialize() #[[INITIALIZE:[0-9]+]]
;
; CHECK-DAG: attributes #[[FINALIZE]] = { "finalize" }
; CHECK-DAG: attributes #[[ID]] = { "id" }
; CHECK-DAG: attributes #[[INITIALIZE]] = { "initialize" }
; CHECK-DAG: attributes #[[THREADS]] = { "threads" }

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]

define void @f(ptr %buf, i64 %n) {
  %numThreads = call i64 @llvm.kit.cpu.num.threads(i32 8) #0
  %threadId = call i64 @llvm.kit.cpu.thread.id(i32 8) #1
  call void @llvm.kit.runtime.finalize(i32 8) #3
  call void @llvm.kit.runtime.initialize(i32 8) #4
  ret void
}

attributes #0 = { "threads" }
attributes #1 = { "id" }
attributes #3 = { "finalize" }
attributes #4 = { "initialize" }
