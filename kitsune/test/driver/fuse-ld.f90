! REQUIRES: kitfc
!
! Kitsune requires that lld built alongside Kitsune be used. As a result,
! -fuse-ld= is not allowed with any value other than lld.
!
! RUN: not %kitfc --tapir=serial -O2 -fuse-ld=bfd -fsyntax-only %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix BFD
!
! RUN: %kitfc --tapir=serial -O2 -fuse-ld=lld -fsyntax-only %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix LLD --allow-empty
!
! BFD: error: unsupported argument 'bfd' to option '-fuse-ld='
! LLD-NOT: {{.+}}
