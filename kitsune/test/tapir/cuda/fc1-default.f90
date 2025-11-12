! REQUIRES: kitfc
!
! ------------------------------------------------------------------------------
! Check that the default options added to the internal command lines (for -fc1
! and the linker) are as expected.
!
! RUN: %kitfc -### --tapir=cuda -O2 %s 2>&1 | FileCheck %s
!
! -fc1 must always get the GPU architecture, virtual architecture and PTX
! version arguments.
!
! CHECK: -fc1
! CHECK-SAME: --tapir=cuda
! CHECK-SAME: --tapir-cuda-arch=sm_{{[0-9]+}}
! CHECK-SAME: --tapir-cuda-virt-arch=compute_{{[0-9]+}}
! CHECK-SAME: --tapir-cuda-features={{[^"]+}}"
! CHECK-SAME: --tapir-cuda-runtime-bc={{[^"]+}}.bc"
!
! CHECK-SAME: --tapir-gpu-prefetch
!
! Stripmining is disabled by default on GPU tapir targets.
!
! CHECK-NOT: -fstripmine
!
! It is a pain to check for the actual linker executable. There are far too
! many options depending on the platform, so just check the next line for the
! expected linker flags.
!
! CHECK-NEXT: -lkitrt
! CHECK-SAME: -lcudart_static
! CHECK-SAME: -lcuda
!
! ------------------------------------------------------------------------------
! Check that the stripmine pass is disabled by default.
!
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 \
! RUN:     --tapir=cuda --tapir-cuda-arch=sm_86 \
! RUN:     -S -emit-llvm %s | FileCheck %s -check-prefix STRIPMINE-PASS
!
! STRIPMINE-PASS-NOT: loop-stripmine
!
! ------------------------------------------------------------------------------

end program
