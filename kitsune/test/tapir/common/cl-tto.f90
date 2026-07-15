! REQUIRES: kitfc
!
! Check that the tapir target options are set correctly depending on the
! frontend options that are passed. These options are common to all tapir
! targets. Since the serial tapir target is guaranteed to be built, we use that
! here.
!
! RUN: %kitfc --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
! RUN:     -mllvm -dump-tapir-target-options \
! RUN:     | FileCheck %s -check-prefixes ALL
!
! RUN: %kitfc --tapir=serial -O1 -S -emit-llvm -o /dev/null %s \
! RUN:     -mllvm -dump-tapir-target-options \
! RUN:     | FileCheck %s --check-prefixes ALL,O1
!
! RUN: %kitfc --tapir=serial -O3 -S -emit-llvm -o /dev/null %s \
! RUN:     -mllvm -dump-tapir-target-options \
! RUN:     | FileCheck %s --check-prefixes ALL,O3
!
! RUN: %kitfc --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
! RUN:     -mllvm -dump-tapir-target-options \
! RUN:     -ffp-contract=off \
! RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
!
! RUN: %kitfc --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
! RUN:     -mllvm -dump-tapir-target-options \
! RUN:     -ffp-contract=fast \
! RUN:     | FileCheck %s -check-prefixes ALL,FP_FAST
!
! ALL:          Tapir target options
! ALL:          Primary: serial
! O1:           Optimization level: O1
! O3:           Optimization level: O3
! FP_STANDARD:  FP fusion: standard
! FP_FAST:      FP fusion: fast
