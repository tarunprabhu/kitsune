! We need a tapir target that supports static linking and which also links
! libkitrt. The serial tapir target is always built, but never links libkitrt.
!
! REQUIRES: kitfc
! REQUIRES: kitsune-opencilk
!
! ----------------------------------------------------------------------------
!
! RUN: %kitfc -### -ftapir=opencilk -O2 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix DYNAMIC
!
! DYNAMIC: -dynamic-linker
! DYNAMIC-SAME: -lkitrt
! DYNAMIC-NOT: -lkitrt_static
!
! ----------------------------------------------------------------------------
!
! RUN: %kitfc -### -ftapir=opencilk -O2 -static-libkitrt %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix LIBKITRT
!
! LIBKITRT: -dynamic-linker
! LIBKITRT-SAME: -Bstatic
! LIBKITRT-SAME: -lkitrt_static
! LIBKITRT-SAME: -Bdynamic
!
! ----------------------------------------------------------------------------
!
! RUN: %kitfc -### -ftapir=opencilk -O2 -static %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STATIC
!
! STATIC: -fc1
! STATIC-NEXT: "-static"
! STATIC-SAME: -lopencilk
! STATIC-SAME: -lkitrt_static
! STATIC-NOT: -Bdynamic

end program
