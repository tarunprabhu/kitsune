! REQUIRES: kitfc

! Flang does not currently handle -print-pipeline-passes. This has been
! implemented upstream, so this test should pass once we upgrade.
! XFAIL: *

! The -fstripmine option is only enabled when the kitsune frontend is used
! with a tapir target
! RUN: not %flang -### -fstripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix FRONTEND
! RUN: not %flang -### -fno-stripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix FRONTEND
! FRONTEND: '-f{{.*}}stripmine' must be used with a Kitsune frontend

! RUN: %kitfc -### -fstripmine %s 2>&1 | FileCheck %s -check-prefix ALLOWED
! RUN: %kitfc -### -fno-stripmine %s 2>&1 | FileCheck %s -check-prefix ALLOWED
! ALLOWED-NOT: must be used with a Kitsune frontend

! Check that the strip mining is enabled correctly depending on the
! optimization level.

! RUN: %kitfc -### -O0 -ftapir=serial %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
! RUN: %kitfc -### -O1 -ftapir=serial %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
! RUN: %kitfc -### -O2 -ftapir=serial %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE
! RUN: %kitfc -### -O3 -ftapir=serial %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE
! RUN: %kitfc -### -O4 -ftapir=serial %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE
! RUN: %kitfc -### -Os -ftapir=serial %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE
! RUN: %kitfc -### -Oz -ftapir=serial %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE

! Check that the -fstripmine and -fno-stripmine flags override the defaults
! RUN: %kitfc -### -O0 -ftapir=serial -fstripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE
! RUN: %kitfc -### -O1 -ftapir=serial -fstripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE
! RUN: %kitfc -### -O2 -ftapir=serial -fno-stripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
! RUN: %kitfc -### -O3 -ftapir=serial -fno-stripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
! RUN: %kitfc -### -O4 -ftapir=serial -fno-stripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
! RUN: %kitfc -### -Os -ftapir=serial -fno-stripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
! RUN: %kitfc -### -Oz -ftapir=serial -fstripmine %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix STRIPMINE

! STRIPMINE: -fstripmine
! NO-STRIPMINE-NOT: -fstripmine

! Check that the stripmine pass is enabled/disabled correctly
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 -fstripmine -ftapir=serial \
! RUN:      -S -emit-llvm %s | FileCheck %s -check-prefix STRIPMINE-PASS
! RUN: %kitfc -mllvm -print-pipeline-passes -O2 -fno-stripmine -ftapir=serial \
! RUN:      -S -emit-llvm %s | FileCheck %s -check-prefix NO-STRIPMINE-PASS

! STRIPMINE-PASS: loop-stripmine
! NO-STRIPMINE-PASS-NOT: loop-stripmine

end program
