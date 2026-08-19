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
; CHECK-NEXT: %[[MALLOCED:.+]] = call noalias ptr @__kitrt_malloc(i64 63) #[[MALLOC:[0-9]+]]
; CHECK-NEXT: call void @__kitrt_free(ptr %[[MALLOCED]]) #[[FREE:[0-9]+]]
;
; CHECK-DAG: attributes #[[ID]] = { "id" }
; CHECK-DAG: attributes #[[THREADS]] = { "threads" }
; CHECK-DAG: attributes #[[MALLOC]] = { "malloc" }
; CHECK-DAG: attributes #[[FREE]] = { "free" }

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]

define void @f(ptr %buf, i64 %n) {
  %numThreads = call i64 @llvm.kit.cpu.num.threads(i32 8) #0
  %threadId = call i64 @llvm.kit.cpu.thread.id(i32 8) #1
  %malloced = call noalias ptr @llvm.kit.cpu.malloc(i32 8, i64 63) #2
  call void @llvm.kit.cpu.free(i32 8, ptr %malloced) #3
  ret void
}

attributes #0 = { "threads" }
attributes #1 = { "id" }
attributes #2 = { "malloc" }
attributes #3 = { "free" }
