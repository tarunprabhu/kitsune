; RUN: opt -passes='lower-mobile-intrinsics' -S -tapir-target=serial %s | FileCheck --check-prefix=SERIAL %s
; RUN: opt -passes='lower-mobile-intrinsics' -S -tapir-target=none %s | FileCheck --check-prefix=NONE %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable
define dso_local void @deallocate(ptr addrspace(67) nocapture noundef readnone %ptr) local_unnamed_addr #0 {
entry:
  tail call void @llvm.kitsune.mobile.free(ptr addrspace(67) %ptr)
  ret void
}

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.kitsune.mobile.free(ptr addrspace(67) nocapture readnone) #1

attributes #0 = { mustprogress nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) }

; SERIAL-LABEL: @deallocate
; SERIAL: call void @__kitrt_default_mem_free(ptr {{.+}})
; SERIAL-NOT: call .+ llvm.kitsune.mobile.free

; NONE-LABEL: @deallocate
; NONE: call void @llvm.kitsune.mobile.free
; NONE-NOT: call .+ @__kitrt_default_mem_free(ptr {{.+}})