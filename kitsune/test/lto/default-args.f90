! REQUIRES: kitfc
!
! -----------------------------------------------------------------------------
!
! Kitsune requires the LLD built alongside Kitsune to be used for LTO. As a
! result, some command line options that allow tweaking the linker may not be
! used with -flto when using Kitsune.
!
! RUN: not %kitfc -### -ftapir=serial -flto -O2 -fuse-ld=lld %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NOT-ALLOWED
!
! NOT-ALLOWED: error: '{{.+}}' cannot be used with -flto in Kitsune
!
! -----------------------------------------------------------------------------
!
! Check that lld is used when LTO is enabled.
!
! RUN: %kitfc -### -ftapir=serial -flto -O2 %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix LINKER-ARGS
!
! LINKER-ARGS: /ld{{(64)?}}.lld
! LINKER-ARGS-SAME: -dynamic-linker
!
! -----------------------------------------------------------------------------
