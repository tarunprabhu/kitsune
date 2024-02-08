// Check that the strip-mining is enabled correctly depending on the
// optimization level.
//
// RUN: %kitxx -### -O1 --tapir=opencilk %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
// RUN: %kitxx -### -O2 --tapir=opencilk %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE
// RUN: %kitxx -### -O3 --tapir=opencilk %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE
// RUN: %kitxx -### -Os --tapir=opencilk %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE
// RUN: not %kitxx -### -Oz --tapir=opencilk %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix ERROR
//
// If strip-mining is only enabled at certain optimization levels, adding
// -fstripmine should have not change the behavior at those levels.
//
// RUN: %kitxx -### -O2 --tapir=opencilk -fstripmine %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix STRIPMINE
// RUN: %kitxx -### -O3 --tapir=opencilk -fstripmine %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix STRIPMINE
// RUN: %kitxx -### -Os --tapir=opencilk -fstripmine %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix STRIPMINE
//
// Check that the -fstripmine and -fno-stripmine flags override the defaults.
//
// RUN: %kitxx -### -O1 --tapir=opencilk -fstripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE
// RUN: %kitxx -### -O2 --tapir=opencilk -fno-stripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
// RUN: %kitxx -### -O3 --tapir=opencilk -fno-stripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
// RUN: %kitxx -### -Os --tapir=opencilk -fno-stripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
// RUN: not %kitxx -### -Oz --tapir=opencilk -fstripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix ERROR
//
// STRIPMINE: -fstripmine
// NO-STRIPMINE-NOT: -fstripmine
//
// ERROR: unsupported optimization level '-Oz'
