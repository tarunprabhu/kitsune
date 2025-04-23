! REQUIRES: kitfc
!
! XFAIL: *
! FIXME:  This currently does not correctly disable strip-mining by default on
! GPU targets because we do not setup the pass manager correctly. This should
! be fixed and the XFAIL removed.
!
! Check that the default options added to the internal command lines (for -fc1
! and the linker) are as expected.
!
! ------------------------------------------------------------------------------
! RUN: %kitfc -### -ftapir=cuda -O2 %s 2>&1 | FileCheck %s
! RUN: %kitfc -### --tapir=cuda -O2 %s 2>&1 | FileCheck %s
!
! -fc1 must always get the GPU architecture, virtual architecture and PTX
! version arguments.
!
! CHECK: -fc1
! CHECK-SAME: --tapir=cuda
! CHECK-SAME: --tapir-cuda-arch=sm_{{[0-9]+}}
! CHECK-SAME: --tapir-cuda-virt-arch=compute_{{[0-9]+}}
! CHECK-SAME: --tapir-cuda-features="{{.+}}"
! CHECK-SAME: --tapir-cuda-runtime-bc="{{.+}}.bc"
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
! Check that the stripmine pass is disabled by default. This checks that the
! the pipeline tuning options object value is set correctly by default.
!
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 -ftapir=cuda \
! RUN:     -S -emit-llvm %s | FileCheck %s -check-prefix STRIPMINE-PASS
!
! STRIPMINE-PASS-NOT: loop-stripmine
!
! ------------------------------------------------------------------------------
! Check that an error is emitted if any of the required options are not
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
!
! ------------------------------------------------------------------------------

end program
