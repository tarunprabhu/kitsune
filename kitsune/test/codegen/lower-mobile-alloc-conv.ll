; Check that the calling convention is preserved when lowering llvm
; .kit.mobile.alloc intrinsics.
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=none -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes NONE %s
;
; NONE: define {{.+}} @allocate(i64 %[[N:.+]])
; NONE-NEXT: %[[PTR1:[0-9]+]] = call fastcc noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %[[N]])
; NONE-NEXT: %[[PTR2:[0-9]+]] = call coldcc noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %[[N]])
; NONE-NEXT: %[[PTR3:[0-9]+]] = call anyregcc noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %[[N]])
; NONE-NEXT: ret ptr addrspace(67) %[[PTR2]]
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes SERIAL %s
;
; SERIAL: define {{.+}} @allocate(i64 %[[N:.+]])
; SERIAL-NEXT: %[[PTR1:[0-9]+]] = call fastcc noalias ptr @malloc(i64 %[[N]])
; SERIAL-NEXT: %[[CST1:[0-9]]] = addrspacecast ptr %[[PTR1]] to ptr addrspace(67)
; SERIAL-NEXT: %[[PTR2:[0-9]+]] = call coldcc noalias ptr @malloc(i64 %[[N]])
; SERIAL-NEXT: %[[CST2:[0-9]]] = addrspacecast ptr %[[PTR2]] to ptr addrspace(67)
; SERIAL-NEXT: %[[PTR3:[0-9]+]] = call anyregcc noalias ptr @malloc(i64 %[[N]])
; SERIAL-NEXT: %[[CST3:[0-9]]] = addrspacecast ptr %[[PTR3]] to ptr addrspace(67)
; SERIAL-NEXT: ret ptr addrspace(67) %[[CST2]]
;
; ------------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

define noalias ptr addrspace(67) @allocate(i64 %n) {
  %1 = call fastcc noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %n)
  %2 = call coldcc noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %n)
  %3 = call anyregcc noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %n)
  ret ptr addrspace(67) %2
}

!0 = !{!"custom-metadata"}
