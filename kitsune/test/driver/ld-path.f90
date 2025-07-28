! REQUIRES: kitfc
!
! XFAIL: *
! NOTE: flang does not currently support --ld-path, so this test will fail
! with an unknown argument error. If flang ever supports --ld-path, this test
! should pass and the XFAIL annotation and this note can be removed.
!
! Kitsune requires that lld built alongside Kitsune be used. As a result,
! options that may override the linker are not allowed.
!
! RUN: not %kitfc --tapir=serial -O2 --ld-path=ld -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s
!
! CHECK: error: '--ld-path=' is not allowed in Kitsune
