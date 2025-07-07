; Check that the llvm.kit.mobile.free intrinsic is lowered correctly for
; various tapir targets.
;
; ------------------------------------------------------------------------------
; RUN: opt -tapir=nolo -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefix=NOLO %s
;
; NOLO: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; NOLO-NEXT: call void @llvm.kit.mobile.free(ptr addrspace(67) %[[P]])
; NOLO-NEXT: ret void
;
; ------------------------------------------------------------------------------
; RUN: opt -tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefix=SERIAL %s
;
; SERIAL: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; SERIAL-NEXT: %[[CST:[0-9]+]] = addrspacecast ptr addrspace(67) %[[P]] to ptr
; SERIAL-NEXT: call void @free(ptr %[[CST]])
; SERIAL-NEXT: ret void
;
; ------------------------------------------------------------------------------
; RUN: %if kitsune-cuda %{ \
; RUN:   opt --tapir=cuda -passes='kit-lower-intrinsics' -S %s \
; RUN:       | FileCheck --check-prefix=CUDA %s \
; RUN: %}
;
; CUDA: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; CUDA-NEXT: %[[CST:[0-9]+]] = addrspacecast ptr addrspace(67) %[[P]] to ptr
; CUDA-NEXT: call void @__kitcuda_mem_free(ptr %[[CST]])
; CUDA-NEXT: ret void
;
; ------------------------------------------------------------------------------
; RUN: %if kitsune-hip %{ \
; RUN:   opt --tapir=hip -passes='kit-lower-intrinsics' -S %s \
; RUN:       | FileCheck --check-prefix=HIP %s \
; RUN: %}
;
; HIP: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; HIP-NEXT: %[[CST:[0-9]+]] = addrspacecast ptr addrspace(67) %[[P]] to ptr
; HIP-NEXT: call void @__kithip_mem_free(ptr %[[CST]])
; HIP-NEXT: ret void
;
; ------------------------------------------------------------------------------
; RUN: %if kitsune-opencilk %{ \
; RUN:   opt --tapir=opencilk -passes='kit-lower-intrinsics' -S %s \
; RUN:       | FileCheck --check-prefix=OPENCILK %s \
; RUN: %}
;
; OPENCILK: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; OPENCILK-NEXT: %[[CST:[0-9]+]] = addrspacecast ptr addrspace(67) %[[P]] to ptr
; OPENCILK-NEXT: call void @free(ptr %[[CST]])
; OPENCILK-NEXT: ret void
;
; ------------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

define void @deallocate(ptr addrspace(67) %p) {
  call void @llvm.kit.mobile.free(ptr addrspace(67) %p)
  ret void
}
