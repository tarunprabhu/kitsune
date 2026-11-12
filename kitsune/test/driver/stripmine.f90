! REQUIRES: kitfc
!
! TODO: -Os and -Oz are not supported in flang at the time of writing these
! tests. Those will eventually be supported, at which time this test should be
! updated to include those as well.
!
! -----------------------------------------------------------------------------
! Check that the strip-mining is enabled correctly depending on the
! optimization level.
!
! RUN: %kitxx -### -O1 --tapir=opencilk %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
! RUN: %kitxx -### -O2 --tapir=opencilk %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE
! RUN: %kitxx -### -O3 --tapir=opencilk %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE
!
! -----------------------------------------------------------------------------
! If strip-mining is only enabled at certain optimization levels, adding
! -fstripmine should not change the behavior at those levels.
!
! RUN: %kitxx -### -O2 --tapir=opencilk -fstripmine %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix STRIPMINE
! RUN: %kitxx -### -O3 --tapir=opencilk -fstripmine %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix STRIPMINE
!
! Check that the -fstripmine and -fno-stripmine flags override the defaults.
!
! -----------------------------------------------------------------------------
! RUN: %kitxx -### -O1 --tapir=opencilk -fstripmine %s 2>&1    \
! RUN:     | FileCheck %s -check-prefix STRIPMINE
! RUN: %kitxx -### -O2 --tapir=opencilk -fno-stripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
! RUN: %kitxx -### -O3 --tapir=opencilk -fno-stripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
!
! -----------------------------------------------------------------------------
! STRIPMINE: -fstripmine
! NO-STRIPMINE-NOT: -fstripmine
!
! ERROR: unsupported optimization level '-O{{[sz]}}'
