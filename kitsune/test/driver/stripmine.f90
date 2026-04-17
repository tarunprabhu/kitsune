! REQUIRES: kitfc
!
!-------------------------------------------------------------------------------
!
! The -fstripmine option is only enabled when the kitsune frontend is used
! with a tapir target
!
! RUN: not %flang -### -fstripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix FRONTEND
! RUN: not %flang -### -fno-stripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix FRONTEND
! FRONTEND: '-f{{.*}}stripmine' must be used with a Kitsune frontend
!
!-------------------------------------------------------------------------------
!
! RUN: %kitfc -### -fstripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefixes=ALLOWED,STRIPMINE
!
! RUN: %kitfc -### -fno-stripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix ALLOWED,NO-STRIPMINE
!
! ALLOWED-NOT: must be used with a Kitsune frontend
! STRIPMINE: -fstripmine
! NO-STRIPMINE-NOT: -fstripmine
!
!-------------------------------------------------------------------------------
! On certain tapir targets, stripmining is enabled by default depending on the
! optimization level. Tests for this behavior are added to the directories
! containing tests for specific tapir targets. These are in
! kitsune/test/tapir/<tt>, where <tt> is a tapir target.
!
!-------------------------------------------------------------------------------
!
! Check that the stripmine pass is enabled/disabled correctly
!
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 -fstripmine --tapir=serial \
! RUN:     -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE-PASS
!
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 -fno-stripmine --tapir=serial \
! RUN:     -S -emit-llvm -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE-PASS
!
! STRIPMINE-PASS: loop-stripmine
! NO-STRIPMINE-PASS-NOT: loop-stripmine
!
!-------------------------------------------------------------------------------
