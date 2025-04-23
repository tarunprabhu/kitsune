! REQUIRES: kitfc
!
! Check that the options required by -fc1 make it to the tapir targets.
!
! RUN: %kitfc -fc1 --tapir=hip --tapir-verbose -O2 %s -o /dev/null \
! RUN:     -emit-llvm \
! RUN:     --tapir-hip-arch=gfx906 \
! RUN:     --tapir-hip-sramecc=off \
! RUN:     --tapir-hip-xnack=on \
! RUN:     --tapir-hip-features="-sramecc:+xnack" \
! RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
! RUN:     --tapir-lld="%S/input/ld.lld" 2>&1 \
! RUN:     | FileCheck %s
!
! CHECK: 'hip' tapir target options
! CHECK: GPU arch: gfx906
! CHECK: SRAMECC: off
! CHECK: Xnack: on
! CHECK: Target features: -sramecc:+xnack
! CHECK: Bitcode files: [
! CHECK:   {{.+}}/amd.bc
! CHECK: ]
! CHECK: LLD: {{.+}}/input/ld.lld

end program
