; Check that the llvm.kit.mobile.alloc intrinsic is lowered correctly for
; various tapir targets.
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=nolo -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefix=NOLO %s
;
; NOLO: define {{.+}} @allocate(i64 %[[N:.+]])
; NOLO-NEXT: %[[PTR:[0-9]+]] = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %[[N]])
; NOLO-NEXT: ret ptr addrspace(67) %[[PTR]]
;
; ------------------------------------------------------------------------------
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefix=SERIAL %s
;
; SERIAL: define {{.+}} @allocate(i64 %[[N:.+]])
; SERIAL-NEXT: %[[PTR:[0-9]+]] = call noalias ptr @malloc(i64 %[[N]])
; SERIAL-NEXT: %[[CST:[0-9]]] = addrspacecast ptr %[[PTR]] to ptr addrspace(67)
; SERIAL-NEXT: ret ptr addrspace(67) %[[CST]]
;
; ------------------------------------------------------------------------------
; RUN: %if kitsune-cuda %{ \
; RUN:   opt --tapir=cuda -passes='kit-lower-intrinsics' -S %s \
; RUN:       | FileCheck --check-prefix=CUDA %s \
; RUN: %}
;
; CUDA: define {{.+}} @allocate(i64 %[[N:.+]])
; CUDA-NEXT: %[[PTR:[0-9]+]] = call noalias ptr @__kitcuda_mem_alloc_managed(i64 %[[N]])
; CUDA-NEXT: %[[CST:[0-9]]] = addrspacecast ptr %[[PTR]] to ptr addrspace(67)
; CUDA-NEXT: ret ptr addrspace(67) %[[CST]]
;
; ------------------------------------------------------------------------------
; RUN: %if kitsune-hip %{ \
; RUN:   opt --tapir=hip -passes='kit-lower-intrinsics' -S %s \
; RUN:       | FileCheck --check-prefix=HIP %s \
; RUN: %}
;
; HIP: define {{.+}} @allocate(i64 %[[N:.+]])
; HIP-NEXT: %[[PTR:[0-9]+]] = call noalias ptr @__kithip_mem_alloc_managed(i64 %[[N]])
; HIP-NEXT: %[[CST:[0-9]]] = addrspacecast ptr %[[PTR]] to ptr addrspace(67)
; HIP-NEXT: ret ptr addrspace(67) %[[CST]]
;
; ------------------------------------------------------------------------------
; RUN: %if kitsune-opencilk %{ \
; RUN:   opt --tapir=opencilk -passes='kit-lower-intrinsics' -S %s \
; RUN:       | FileCheck --check-prefix=OPENCILK %s \
; RUN: %}
;
; OPENCILK: define {{.+}} @allocate(i64 %[[N:.+]])
; OPENCILK-NEXT: %[[PTR:[0-9]+]] = call noalias ptr @malloc(i64 %[[N]])
; OPENCILK-NEXT: %[[CST:[0-9]]] = addrspacecast ptr %[[PTR]] to ptr addrspace(67)
; OPENCILK-NEXT: ret ptr addrspace(67) %[[CST]]
;
; ------------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

define noalias ptr addrspace(67) @allocate(i64 %n) {
  %1 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %n)
  ret ptr addrspace(67) %1
}
