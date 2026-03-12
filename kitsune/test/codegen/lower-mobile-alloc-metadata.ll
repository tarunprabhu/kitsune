; Check that metadata is preserved when lowering llvm.kit.mobile.alloc
; intrinsics.
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=nolo -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes NOLO,METADATA %s
;
; NOLO: define {{.+}} @allocate(i64 %[[N:.+]])
; NOLO-NEXT: %[[PTR:[0-9]+]] = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %[[N]]), !custom-key ![[MD:[0-9]+]]
; NOLO-NEXT: ret ptr addrspace(67) %[[PTR]]
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes SERIAL,METADATA %s
;
; SERIAL: define {{.+}} @allocate(i64 %[[N:.+]])
; SERIAL-NEXT: %[[PTR:[0-9]+]] = call noalias ptr @malloc(i64 %[[N]]), !custom-key ![[MD:[0-9]+]]
; SERIAL-NEXT: %[[CST:[0-9]]] = addrspacecast ptr %[[PTR]] to ptr addrspace(67)
; SERIAL-NEXT: ret ptr addrspace(67) %[[CST]]
;
; ------------------------------------------------------------------------------
;
; METADATA: ![[MD]] = !{!"custom-metadata"}
;
; ------------------------------------------------------------------------------

target triple = "x86_64-pc-linux-gnu"

define noalias ptr addrspace(67) @allocate(i64 %n) {
  %1 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %n), !custom-key !0
  ret ptr addrspace(67) %1
}

!0 = !{!"custom-metadata"}
