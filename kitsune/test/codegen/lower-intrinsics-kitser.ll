; Check that intrinsics that map to Kitsune's serial runtime are lowered
; correctly. If more intrinsics are created, they should be added here to test
; basic intrinsic lowering.
;
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: call void @__kitser_initialize() #[[INITIALIZE:[0-9]+]]
; CHECK-NEXT: call i64 @__kitser_thread_id() #[[ID:[0-9]+]]
; CHECK-NEXT: call void @__kitser_finalize() #[[FINALIZE:[0-9]+]]
;
; CHECK-DAG: attributes #[[INITIALIZE]] = { "initialize" }
; CHECK-DAG: attributes #[[FINALIZE]] = { "finalize" }
; CHECK-DAG: attributes #[[ID]] = { "id" }

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

define void @f(ptr %buf, i64 %n) {
  call void @llvm.kit.runtime.initialize(i32 1) #0
  %threadID = call i64 @llvm.kit.cpu.thread.id(i32 1) #2
  call void @llvm.kit.runtime.finalize(i32 1) #1
  ret void
}

attributes #0 = { "initialize" }
attributes #1 = { "finalize" }
attributes #2 = { "id" }
