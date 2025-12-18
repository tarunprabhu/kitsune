! REQUIRES: kitfc
!
! If both kitfc.cfg and flang.cfg are present in the same directory, ensure
! that the correct configuration file is read.
!
! RUN: env CLANG_NO_DEFAULT_CONFIG= \
! RUN:     not %kitfc --config-system-dir=%S/input/cfgs %s 2>&1 \
! RUN:         | FileCheck %s --check-prefixes=KITFC
!
! RUN: env CLANG_NO_DEFAULT_CONFIG= \
! RUN:     not %kitfc --config-user-dir=%S/input/cfgs %s 2>&1 \
! RUN:         | FileCheck %s --check-prefixes=KITFC
!
! KITFC: error: unknown argument: '--not-a-kitfc-option'
!
! RUN: env CLANG_NO_DEFAULT_CONFIG= \
! RUN:     not %flang --config-system-dir=%S/input/cfgs %s 2>&1 \
! RUN:         | FileCheck %s --check-prefixes=FLANG
!
! RUN: env CLANG_NO_DEFAULT_CONFIG= \
! RUN:     not %flang --config-user-dir=%S/input/cfgs %s 2>&1 \
! RUN:         | FileCheck %s --check-prefixes=FLANG
!
! FLANG: error: unknown argument: '--not-a-flang-option'
