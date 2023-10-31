; RUN: not llvm-as -o /dev/null %s 2>&1 | FileCheck %s

declare void @llvm.memory.used(ptr, i32, i32, i64, i64)

define void @use_range(ptr %ptr, i64 %beg, i64 %end) {
  ; CHECK: Argument #2 (use) to llvm.memory.used must be in [0, 4).
  ; CHECK-NEXT: call void @llvm.memory.used.p0(ptr %ptr, i32 -1, i32 0, i64 %beg, i64 %end)
  ; CHECK: Argument #2 (use) to llvm.memory.used must be in [0, 4).
  ; CHECK-NEXT: call void @llvm.memory.used.p0(ptr %ptr, i32 4, i32 0, i64 %beg, i64 %end)
  call void @llvm.memory.used(ptr %ptr, i32 -1, i32 0, i64 %beg, i64 %end)
  call void @llvm.memory.used(ptr %ptr, i32 4, i32 0, i64 %beg, i64 %end)
  call void @llvm.memory.used(ptr %ptr, i32 0, i32 0, i64 %beg, i64 %end)
  call void @llvm.memory.used(ptr %ptr, i32 1, i32 0, i64 %beg, i64 %end) 
  call void @llvm.memory.used(ptr %ptr, i32 2, i32 0, i64 %beg, i64 %end)
  call void @llvm.memory.used(ptr %ptr, i32 3, i32 0, i64 %beg, i64 %end)

  ret void
}

define void @where_range(ptr %ptr) {
  ; CHECK: Argument #3 (where) to llvm.memory.used must be >= 0.
  ; CHECK-NEXT: call void @llvm.memory.used.p0(ptr %ptr, i32 0, i32 -1, i64 -1, i64 -1)
  call void @llvm.memory.used(ptr %ptr, i32 0, i32 -1, i64 -1, i64 -1)
  call void @llvm.memory.used(ptr %ptr, i32 0, i32 0, i64 -1, i64 -1)
  call void @llvm.memory.used(ptr %ptr, i32 0, i32 1, i64 -1, i64 -1)

  ret void
}

define void @beg_range(ptr %ptr) {
  ; CHECK: Argument #4 (begin) to llvm.memory.used must be >= -1.
  ; CHECK-NEXT: call void @llvm.memory.used.p0(ptr %ptr, i32 0, i32 0, i64 -2, i64 -1)
  call void @llvm.memory.used(ptr %ptr, i32 0, i32 0, i64 -2, i64 -1)
  call void @llvm.memory.used(ptr %ptr, i32 0, i32 0, i64 -1, i64 -1)
  call void @llvm.memory.used(ptr %ptr, i32 0, i32 0, i64 0, i64 -1)

  ret void
}

define void @end_range(ptr %ptr) {
  ; CHECK: Argument #5 (end) to llvm.memory.used must be >= -1.
  ; CHECK-NEXT: call void @llvm.memory.used.p0(ptr %ptr, i32 0, i32 0, i64 -1, i64 -2)
  call void @llvm.memory.used(ptr %ptr, i32 0, i32 0, i64 -1, i64 -2)
  call void @llvm.memory.used(ptr %ptr, i32 0, i32 0, i64 -1, i64 -1)
  call void @llvm.memory.used(ptr %ptr, i32 0, i32 0, i64 -1, i64 0)

  ret void
}
