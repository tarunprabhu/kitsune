! REQUIRES: kitfc
!
! ------------------------------------------------------------------------------
! Check that an error is emitted if any of the required options are not
! provided
!
! RUN: not %kitfc -fc1 --tapir=hip %s -o /dev/null \
! RUN:     --tapir-lld="%S/input/ld.lld" 2>&1 \
! RUN:     --tapir-hip-sramecc=any \
! RUN:     --tapir-hip-xnack=any \
! RUN:     --tapir-hip-features="+16-bit-insts" \
! RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" 2>&1 \
! RUN:     | FileCheck %s -check-prefix=MISSING_ARCH
!
! RUN: not %kitfc -fc1 --tapir=hip %s -o /dev/null \
! RUN:     --tapir-lld="%S/input/ld.lld" 2>&1 \
! RUN:     --tapir-hip-arch=gfx90a \
! RUN:     --tapir-hip-xnack=any \
! RUN:     --tapir-hip-features="+16-bit-insts" \
! RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" 2>&1 \
! RUN:     | FileCheck %s -check-prefix=MISSING_SRAMECC
!
! RUN: not %kitfc -fc1 --tapir=hip %s -o /dev/null \
! RUN:     --tapir-lld="%S/input/ld.lld" 2>&1 \
! RUN:     --tapir-hip-arch=gfx90a \
! RUN:     --tapir-hip-sramecc=any \
! RUN:     --tapir-hip-features="+16-bit-insts" \
! RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" 2>&1 \
! RUN:     | FileCheck %s -check-prefix=MISSING_XNACK
!
! RUN: not %kitfc -fc1 --tapir=hip %s -o /dev/null \
! RUN:     --tapir-lld="%S/input/ld.lld" 2>&1 \
! RUN:     --tapir-hip-arch=gfx90a \
! RUN:     --tapir-hip-sramecc=any \
! RUN:     --tapir-hip-xnack=any \
! RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" 2>&1 \
! RUN:     | FileCheck %s -check-prefix=MISSING_FEATURES
!
! RUN: not %kitfc -fc1 --tapir=hip %s -o /dev/null \
! RUN:     --tapir-lld="%S/input/ld.lld" 2>&1 \
! RUN:     --tapir-hip-arch=gfx90a \
! RUN:     --tapir-hip-sramecc=any \
! RUN:     --tapir-hip-xnack=any \
! RUN:     --tapir-hip-features="+16-bit-insts" \
! RUN:     | FileCheck %s -check-prefix=MISSING_RUNTIME_BCS
!
! RUN: not %kitfc -fc1 --tapir=hip %s -o /dev/null \
! RUN:     --tapir-hip-arch=gfx90a \
! RUN:     --tapir-hip-sramecc=any \
! RUN:     --tapir-hip-xnack=any \
! RUN:     --tapir-hip-features="+16-bit-insts" \
! RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" 2>&1 \
! RUN:     | FileCheck %s -check-prefix=MISSING_LLD
!
! MISSING_ARCH: missing required option '--tapir-hip-arch='
! MISSING_SRAMECC: missing required option '--tapir-hip-sramecc='
! MISSING_XNACK: missing required option '--tapir-hip-xnack='
! MISSING_FEATURES: missing required option '--tapir-hip-features='
! MISSING_RUNTIME_BCS: missing required option '--tapir-hip-runtime-bcs='
! MISSING_LLD: missing required option '--tapir-lld='
!
! ------------------------------------------------------------------------------

end program
