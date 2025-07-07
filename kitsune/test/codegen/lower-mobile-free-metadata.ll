; Check that metadata is preserved when lowering the llvm.kit.mobile.free
; intrinsic
;
; ------------------------------------------------------------------------------
; RUN: opt -tapir=nolo -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes NOLO,METADATA %s
;
; NOLO: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; NOLO-NEXT: call void @llvm.kit.mobile.free(ptr addrspace(67) %[[P]]), !custom-key ![[MD:[0-9]+]]
; NOLO-NEXT: ret void
;
; ------------------------------------------------------------------------------
; RUN: opt -tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefixes SERIAL,METADATA %s
;
; SERIAL: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; SERIAL-NEXT: %[[CST:[0-9]+]] = addrspacecast ptr addrspace(67) %[[P]] to ptr
; SERIAL-NEXT: call void @free(ptr %[[CST]]), !custom-key ![[MD:[0-9]+]]
; SERIAL-NEXT: ret void
;
; ------------------------------------------------------------------------------
;
; METADATA: ![[MD]] = !{!"custom-metadata"}
;
; ------------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

define void @deallocate(ptr addrspace(67) %p) {
  call void @llvm.kit.mobile.free(ptr addrspace(67) %p), !custom-key !0
  ret void
}

!0 = !{!"custom-metadata"}
