// Kitsune supports only a subset of the optimization levels that clang does.
// Check that Kitsune errors out, and also does not emit clang's warnings.
//
// RUN: %kitxx -### -O0 %s 2>&1 | FileCheck %s -check-prefix O0
// RUN: %kitxx -### -O1 %s 2>&1 | FileCheck %s -check-prefix O1
// RUN: %kitxx -### -O2 %s 2>&1 | FileCheck %s -check-prefix O2
// RUN: %kitxx -### -O3 %s 2>&1 | FileCheck %s -check-prefix O3
// RUN: %kitxx -### -Os %s 2>&1 | FileCheck %s -check-prefix OS
// RUN: not %kitxx -### -Oz %s 2>&1 | FileCheck %s -check-prefixes ERROR,OZ
// RUN: not %kitxx -### -O4 %s 2>&1 | FileCheck %s -check-prefixes ERROR,O4
// RUN: not %kitxx -### -O5 %s 2>&1 | FileCheck %s -check-prefixes ERROR,O5
// RUN: not %kitxx -### -Ofast %s 2>&1 | FileCheck %s -check-prefixes ERROR,FAST
//
// O0: -O0
// O1: -O1
// O2: -O2
// O3: -O3
// OS: -Os
// O4-NOT: -O4 is equivalent to -O3
// O5-NOT: optimization level {{.+}} is not supported
// FAST-NOT: argument '-Ofast' is deprecated
// ERROR: unsupported optimization level
// OZ: -Oz
