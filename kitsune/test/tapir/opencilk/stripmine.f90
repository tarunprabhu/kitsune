! REQUIRES: kitfc
!
! Check that the strip-mining is enabled correctly depending on the optimization
! level.
!
! RUN: %kitfc -### -O1 --tapir=opencilk %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
! RUN: %kitfc -### -O2 --tapir=opencilk %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE
! RUN: %kitfc -### -O3 --tapir=opencilk %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE
! RUN: %kitfc -### -Os --tapir=opencilk %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE
! RUN: not %kitfc -### -Oz --tapir=opencilk %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix ERROR
!
! If strip-mining is only enabled at certain optimization levels, adding
! -fstripmine should have not change the behavior at those levels.
!
! RUN: %kitfc -### -O2 --tapir=opencilk -fstripmine %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix STRIPMINE
! RUN: %kitfc -### -O3 --tapir=opencilk -fstripmine %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix STRIPMINE
! RUN: %kitfc -### -Os --tapir=opencilk -fstripmine %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix STRIPMINE
!
! Check that the -fstripmine and -fno-stripmine flags override the defaults.
!
! RUN: %kitfc -### -O1 --tapir=opencilk -fstripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE
! RUN: %kitfc -### -O2 --tapir=opencilk -fno-stripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
! RUN: %kitfc -### -O3 --tapir=opencilk -fno-stripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
! RUN: %kitfc -### -Os --tapir=opencilk -fno-stripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
! RUN: not %kitfc -### -Oz --tapir=opencilk -fstripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix ERROR
!
! STRIPMINE: -fstripmine
! NO-STRIPMINE-NOT: -fstripmine
!
! ERROR: unsupported optimization level '-Oz'
