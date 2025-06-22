! REQUIRES: kitfc
!
! ------------------------------------------------------------------------------
!
! Check that an error is emitted if any of the options required by fc1 are not
! provided.
!
! RUN: not %kitfc -fc1 --tapir=cuda %s -o /dev/null \
! RUN:     --tapir-cuda-virt-arch=compute_72 \
! RUN:     --tapir-cuda-features="+ptx72" \
! RUN:     --tapir-cuda-runtime-bc="%S/input/nvptx.bc" 2>&1 \
! RUN:     | FileCheck %s -check-prefix=MISSING_ARCH
!
! RUN: not %kitfc -fc1 --tapir=cuda %s -o /dev/null \
! RUN:     --tapir-cuda-arch=sm_72 \
! RUN:     --tapir-cuda-features="+ptx72" \
! RUN:     --tapir-cuda-runtime-bc="%S/input/nvptx.bc" 2>&1 \
! RUN:     | FileCheck %s -check-prefix=MISSING_VIRTARCH
!
! RUN: not %kitfc -fc1 --tapir=cuda %s -o /dev/null \
! RUN:     --tapir-cuda-arch=sm_72 \
! RUN:     --tapir-cuda-virt-arch=compute_72 \
! RUN:     --tapir-cuda-runtime-bc="%S/input/nvptx.bc" 2>&1 \
! RUN:     | FileCheck %s -check-prefix=MISSING_FEATURES
!
! RUN: not %kitfc -fc1 --tapir=cuda %s -o /dev/null \
! RUN:     --tapir-cuda-arch=sm_72 \
! RUN:     --tapir-cuda-virt-arch=compute_72 \
! RUN:     --tapir-cuda-features="+ptx72" 2>&1 \
! RUN:     | FileCheck %s -check-prefix=MISSING_RUNTIME_BC
!
! MISSING_ARCH: missing required option '--tapir-cuda-arch='
! MISSING_VIRTARCH: missing required option '--tapir-cuda-virt-arch='
! MISSING_FEATURES: missing required option '--tapir-cuda-features='
! MISSING_RUNTIME_BC: missing required option '--tapir-cuda-runtime-bc='

end program
