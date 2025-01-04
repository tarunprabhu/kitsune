; RUN: opt -passes='lower-mobile-intrinsics' -S -tapir-target=serial %s | FileCheck --check-prefix=SERIAL %s
; RUN: opt -passes='lower-mobile-intrinsics' -S -tapir-target=none %s | FileCheck --check-prefix=NONE %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind memory(inaccessiblemem: readwrite) uwtable
define dso_local noalias ptr addrspace(67) @allocate(i64 noundef %n) local_unnamed_addr #0 {
entry:
  %mul = shl i64 %n, 2
  %0 = tail call noalias ptr addrspace(67) @llvm.kitsune.mobile.alloc(i64 %mul)
  ret ptr addrspace(67) %0
}

; Function Attrs: nounwind memory(inaccessiblemem: readwrite)
declare noalias ptr addrspace(67) @llvm.kitsune.mobile.alloc(i64) #1

attributes #0 = { nounwind memory(inaccessiblemem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind memory(inaccessiblemem: readwrite) }

; SERIAL-LABEL: @allocate
; SERIAL: call noalias ptr addrspace(67) @__kitrt_default_mem_alloc(i64 %mul)
; SERIAL-NOT: call .+ llvm.kitsune.mobile.alloc
; SERIAL-DAG: declare noalias ptr addrspace(67) @__kitrt_default_mem_alloc(i64)

; NONE-LABEL: @allocate
; NONE: call noalias ptr addrspace(67) @llvm.kitsune.mobile.alloc
; NONE-NOT: call .+ @__kitrt_default_mem_alloc(%mul)