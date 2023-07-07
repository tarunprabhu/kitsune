; RUN: not llvm-as -o /dev/null %s 2>&1 | FileCheck %s

declare void @llvm.asyncprefetch(ptr, i32, i64)

define void @rw_range(ptr %ptr) {
  ; CHECK: Argument #2 (rw) to llvm.asyncprefetch must be in [1, 3].
  ; CHECK-NEXT: call void @llvm.asyncprefetch.p0(ptr %ptr, i32 0, i64 32)
  ; CHECK: Argument #2 (rw) to llvm.asyncprefetch must be in [1, 3].
  ; CHECK-NEXT: call void @llvm.asyncprefetch.p0(ptr %ptr, i32 4, i64 32)
  call void @llvm.asyncprefetch(ptr %ptr, i32 0, i64 32)
  call void @llvm.asyncprefetch(ptr %ptr, i32 1, i64 32)
  call void @llvm.asyncprefetch(ptr %ptr, i32 2, i64 32)
  call void @llvm.asyncprefetch(ptr %ptr, i32 3, i64 32)
  call void @llvm.asyncprefetch(ptr %ptr, i32 4, i64 32)

  ret void
}

define void @size_range(ptr %ptr, i64 %size) {
  ; CHECK: Argument #3 (size) to llvm.asyncprefetch must be >= 0.
  ; CHECK-NEXT: call void @llvm.asyncprefetch.p0(ptr %ptr, i32 1, i64 -1)
  call void @llvm.asyncprefetch(ptr %ptr, i32 1, i64 -1)
  call void @llvm.asyncprefetch(ptr %ptr, i32 2, i64 0)
  call void @llvm.asyncprefetch(ptr %ptr, i32 3, i64 1)
  call void @llvm.asyncprefetch(ptr %ptr, i32 3, i64 %size)

  ret void
}