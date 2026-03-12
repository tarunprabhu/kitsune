; Check that the llvm.kit.mobile.free intrinsic is lowered correctly for
; various tapir targets.
;
; ------------------------------------------------------------------------------
; When the tapir target is 'nolo', the intrinsic is not lowered.
;
; RUN: opt -tapir=nolo -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefix=NOLO %s
;
; NOLO: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; NOLO-NEXT: call void @llvm.kit.mobile.free(ptr addrspace(67) %[[P]])
; NOLO-NEXT: ret void
;
; ------------------------------------------------------------------------------
; When the tapir target is 'cuda', call the appropriate function from Kitsune's
; runtime.
;
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
; When the tapir target is 'hip', call the appropriate function from Kitsune's
; runtime.
;
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
; For all other tapir targets, call the default memory deallocation function
; from libc.
;
; RUN: %if kitsune-examples %{ \
; RUN:   opt --tapir=custom --tapir-plugin=%kit-tt-plugin-demo \
; RUN:       -passes='kit-lower-intrinsics' -S %s \
; RUN:       | FileCheck --check-prefix=FREE %s \
; RUN: %}
;
; RUN: %if kitsune-opencilk %{ \
; RUN:   opt --tapir=opencilk -passes='kit-lower-intrinsics' -S %s \
; RUN:       | FileCheck --check-prefix=FREE %s \
; RUN: %}
;
; RUN: opt --tapir=pthreads -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefix=FREE %s
;
; RUN: %if kitsune-qthreads %{ \
; RUN:   opt --tapir=qthreads -passes='kit-lower-intrinsics' -S %s \
; RUN:       | FileCheck --check-prefix=FREE %s \
; RUN: %}
;
; RUN: opt -tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefix=FREE %s
;
; FREE: define {{.+}} @deallocate(ptr addrspace(67) %[[P:.+]])
; FREE-NEXT: %[[CST:[0-9]+]] = addrspacecast ptr addrspace(67) %[[P]] to ptr
; FREE-NEXT: call void @free(ptr %[[CST]])
; FREE-NEXT: ret void
;
; ------------------------------------------------------------------------------

target triple = "x86_64-pc-linux-gnu"

define void @deallocate(ptr addrspace(67) %p) {
  call void @llvm.kit.mobile.free(ptr addrspace(67) %p)
  ret void
}
