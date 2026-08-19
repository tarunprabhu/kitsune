; Check that intrinsics called with the serial tapir target are lowered
; correctly.
;
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: %[[MALLOCED:.+]] = call noalias ptr @__kitrt_malloc(i64 63) #[[MALLOC:[0-9]+]]
; CHECK-NEXT: call void @__kitrt_free(ptr %[[MALLOCED]]) #[[FREE:[0-9]+]]

; CHECK-DAG: attributes #[[MALLOC]] = { "malloc" }
; CHECK-DAG: attributes #[[FREE]] = { "free" }

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

define void @f(ptr %buf, i64 %n) {
  %malloced = call noalias ptr @llvm.kit.cpu.malloc(i32 1, i64 63) #0
  call void @llvm.kit.cpu.free(i32 1, ptr %malloced) #1
  ret void
}

attributes #0 = { "malloc" }
attributes #1 = { "free" }
