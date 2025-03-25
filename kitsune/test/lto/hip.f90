! REQUIRES: kitfc
! REQUIRES: kitsune-hip
!
! RUN: %kitfc -### -ftapir=hip -O2 -flto %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL
!
! RUN: %kitfc -### -ftapir=hip -O2 -flto %s \
! RUN:     --tapir-verbose 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL,TAPIR-VERBOSE
!
! RUN: %kitfc -### -ftapir=hip -O2 -flto %s \
! RUN:     --kitrt-verbose 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL,KITRT-VERBOSE
!
! RUN: %kitfc -### -ftapir=hip -O2 -flto %s \
! RUN:     --tapir-hip-arch=gfx906 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL,HIP_ARCH
!
! RUN: %kitfc -### -ftapir=hip -O2 -flto %s \
! RUN:     --tapir-threads-per-block=64 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL,TPB
!
! RUN: %kitfc -### -ftapir=hip -O2 -flto %s \
! RUN:     --tapir-max-threads-per-block=128 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALL,MTPB

! lld will be present in the compile line as well because the path to it will
! be a fc1 argument. First look for fc1, then the next line which should be the
! link line
!
! ALL: -fc1
! ALL-SAME: /ld{{(64)?}}.lld"
! ALL-NEXT: /ld{{(64)?}}.lld"
! TAPIR-VERBOSE: --tapir-verbose
! KITRT-VERBOSE: --kitrt-verbose
! ALL-SAME: --tapir=hip
! HIP_ARCH: --tapir-hip-arch=gfx906
! TPB: --tapir-threads-per-block=64
! MTPB: --tapir-max-threads-per-block=128

end program
