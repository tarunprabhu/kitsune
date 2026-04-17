// -----------------------------------------------------------------------------
//
// The -fstripmine option is only enabled when the kitsune frontend is used
// with a tapir target
//
// RUN: not %clang -### -fstripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix FRONTEND
// RUN: not %clang -### -fno-stripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix FRONTEND
// FRONTEND: '-f{{.*}}stripmine' must be used with a Kitsune frontend
//
// -----------------------------------------------------------------------------
//
// RUN: %kitxx -### -fstripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=ALLOWED,STRIPMINE
//
// RUN: %kitxx -### -fno-stripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=ALLOWED,NO-STRIPMINE
//
// ALLOWED-NOT: must be used with a Kitsune frontend
// STRIPMINE: -fstripmine
// NO-STRIPMINE-NOT: -fstripmine
//
// -----------------------------------------------------------------------------
// On certain tapir targets, stripmining is enabled by default depending on the
// optimization level. Tests for this behavior are added to the directories
// containing tests for specific tapir targets. These are in
// kitsune/test/tapir/<tt>, where <tt> is a tapir target.
//
// -----------------------------------------------------------------------------
