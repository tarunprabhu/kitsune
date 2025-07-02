! REQUIRES: kitfc
!
! Kitsune supports only a subset of the optimization levels that flang does.
! Check that Kitsune errors out, and also does not emit flang's warnings.
!
! RUN: %kitfc -### -O0 %s 2>&1 | FileCheck %s -check-prefix O0
! RUN: %kitfc -### -O1 %s 2>&1 | FileCheck %s -check-prefix O1
! RUN: %kitfc -### -O2 %s 2>&1 | FileCheck %s -check-prefix O2
! RUN: %kitfc -### -O3 %s 2>&1 | FileCheck %s -check-prefix O3
! RUN: %kitfc -### -Os %s 2>&1 | FileCheck %s -check-prefix OS
! RUN: %kitfc -### -Oz %s 2>&1 | FileCheck %s -check-prefix OZ
! RUN: not %kitfc -### -O4 %s 2>&1 | FileCheck %s -check-prefixes ERROR,O4
! RUN: not %kitfc -### -O5 %s 2>&1 | FileCheck %s -check-prefixes ERROR,O5
! RUN: not %kitfc -### -Ofast %s 2>&1 | FileCheck %s -check-prefixes ERROR,FAST
!
! O0: -O0
! O1: -O1
! O2: -O2
! O3: -O3
! OS: -Os
! OZ: -Oz
! O4-NOT: -O4 is equivalent to -O3
! O5-NOT: optimization level {{.+}} is not supported
! FAST-NOT: argument '-Ofast' is deprecated
! ERROR: unsupported optimization level
