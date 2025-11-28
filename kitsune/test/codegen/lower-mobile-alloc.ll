; Check that the llvm.kit.mobile.alloc intrinsic is lowered correctly for
; various tapir targets.
;
; ------------------------------------------------------------------------------
; When the tapir target is 'nolo', the intrinsic is not lowered.
;
; RUN: opt --tapir=nolo -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefix=NOLO %s
;
; NOLO: define {{.+}} @allocate(i64 %[[N:.+]])
; NOLO-NEXT: %[[PTR:[0-9]+]] = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %[[N]])
; NOLO-NEXT: ret ptr addrspace(67) %[[PTR]]
;
; ------------------------------------------------------------------------------
; When the tapir target is 'cuda', call the appropriate function from Kitsune's
; runtime.
;
; RUN: %if kitsune-cuda %{ \
; RUN:   opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:       --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:       -passes='kit-lower-intrinsics' -S %s \
; RUN:       | FileCheck --check-prefix=CUDA %s \
; RUN: %}
;
; CUDA: define {{.+}} @allocate(i64 %[[N:.+]])
; CUDA-NEXT: %[[PTR:[0-9]+]] = call noalias ptr @__kitcuda_mem_alloc_managed(i64 %[[N]])
; CUDA-NEXT: %[[CST:[0-9]]] = addrspacecast ptr %[[PTR]] to ptr addrspace(67)
; CUDA-NEXT: ret ptr addrspace(67) %[[CST]]
;
; ------------------------------------------------------------------------------
; When the tapir target is 'hip', call the appropriate function from Kitsune's
; runtime.
;
; RUN: %if kitsune-hip %{ \
; RUN:   opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:       --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:       -passes='kit-lower-intrinsics' -S %s \
; RUN:       | FileCheck --check-prefix=HIP %s \
; RUN: %}
;
; HIP: define {{.+}} @allocate(i64 %[[N:.+]])
; HIP-NEXT: %[[PTR:[0-9]+]] = call noalias ptr @__kithip_mem_alloc_managed(i64 %[[N]])
; HIP-NEXT: %[[CST:[0-9]]] = addrspacecast ptr %[[PTR]] to ptr addrspace(67)
; HIP-NEXT: ret ptr addrspace(67) %[[CST]]
;
; ------------------------------------------------------------------------------
; For all other tapir targets, call the default memory allocation function from
; libc.
;
; RUN: %if kitsune-examples %{ \
; RUN:   opt --tapir=custom --tapir-plugin=%kit-tt-plugin-demo \
; RUN:       -passes='kit-lower-intrinsics' -S %s \
; RUN:       | FileCheck --check-prefix=MALLOC %s \
; RUN: %}
;
; RUN: %if kitsune-opencilk %{ \
; RUN:   opt --tapir=opencilk \
; RUN:       --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
; RUN:       -passes='kit-lower-intrinsics' -S %s \
; RUN:       | FileCheck --check-prefix=MALLOC %s \
; RUN: %}
;
; RUN: opt --tapir=pthreads -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefix=MALLOC %s
;
; RUN: opt --tapir=serial -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck --check-prefix=MALLOC %s
;
; MALLOC: define {{.+}} @allocate(i64 %[[N:.+]])
; MALLOC-NEXT: %[[PTR:[0-9]+]] = call noalias ptr @malloc(i64 %[[N]])
; MALLOC-NEXT: %[[CST:[0-9]]] = addrspacecast ptr %[[PTR]] to ptr addrspace(67)
; MALLOC-NEXT: ret ptr addrspace(67) %[[CST]]
;
; ------------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

define noalias ptr addrspace(67) @allocate(i64 %n) {
  %1 = call noalias ptr addrspace(67) @llvm.kit.mobile.alloc(i64 %n)
  ret ptr addrspace(67) %1
}
