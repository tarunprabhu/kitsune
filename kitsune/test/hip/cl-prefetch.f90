! REQUIRES: kitfc
!
! Check that the --tapir-gpu-prefetch and --tapir-gpu-no-prefetch command line
! options are handled correctly.
!
! RUN: %kitfc -### --tapir=hip -S -emit-llvm -O2 -o - %s \
! RUN:     --tapir-hip-arch=gfx90a 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,PREFETCH
!
! RUN: %kitfc -### --tapir=hip -S -emit-llvm -O2 -o - %s \
! RUN:     --tapir-gpu-prefetch \
! RUN:     --tapir-hip-arch=gfx90a 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,PREFETCH
!
! RUN: %kitfc -### --tapir=hip -S -emit-llvm -O2 -o - %s \
! RUN:     --tapir-gpu-no-prefetch \
! RUN:     --tapir-hip-arch=gfx90a 2>&1 \
! RUN:     | FileCheck %s -check-prefixes ALL,NO-PREFETCH
!
! ALL: -fc1
! PREFETCH: --tapir-gpu-prefetch
! NO-PREFETCH: --tapir-gpu-no-prefetch
