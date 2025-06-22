! REQUIRES: kitfc
!
! OpenMP offload cannot be used together with a Kitsune tapir target.
!
! RUN: not %kitfc -### %s -c -O2  \
! RUN:     --tapir=serial -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu 2>&1 \
! RUN:     | FileCheck %s -check-prefix BAD
!
! Running the kitsune frontend without --tapir is ok
!
! RUN: %kitfc -### %s -c -O2  \
! RUN:     -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu 2>&1 \
! RUN:     | FileCheck %s -check-prefix OK
!
! BAD: cannot use OpenMP offload with a tapir target
! OK-NOT: cannot use OpenMP offload with a tapir target
