! REQUIRES: kitfc
!
! XFAIL: *
! FIXME: This has not been implemented
!
! Check that the options required by -fc1 make it to the tapir targets.
!
! RUN: %kitfc -fc1 --tapir=cuda --tapir-verbose -O2 %s -o /dev/null \
! RUN:     -disable-free -emit-llvm \
! RUN:     --tapir-cuda-arch=sm_72 \
! RUN:     --tapir-cuda-virt-arch=compute_72 \
! RUN:     --tapir-cuda-features="+ptx72" \
! RUN:     --tapir-cuda-runtime-bc="%S/input/nvptx.bc" 2>&1 \
! RUN:     | FileCheck %s
!
! CHECK: 'cuda' tapir target options
! CHECK: GPU arch: sm_72
! CHECK: GPU virtual arch: compute_72
! CHECK: Target features: +ptx72
! CHECK: Bitcode file: {{.+}}.bc
!
! FIXME: We need a DO CONCURRENT loop so CudaABI is entered.
