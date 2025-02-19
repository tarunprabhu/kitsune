// The -fstripmine option is only enabled when the kitsune frontend is used
// with a tapir target
// RUN: not %clang -### -fstripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix FRONTEND
// RUN: not %clang -### -fno-stripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix FRONTEND
// FRONTEND: '-f{{.*}}stripmine' must be used with a Kitsune frontend

// RUN: %kitxx -### -fstripmine %s 2>&1 | FileCheck %s -check-prefix ALLOWED
// RUN: %kitxx -### -fno-stripmine %s 2>&1 | FileCheck %s -check-prefix ALLOWED
// ALLOWED-NOT: must be used with a Kitsune frontend

// Check that the strip mining is enabled correctly depending on the
// optimization level.

// RUN: %kitxx -### -O0 -ftapir=serial %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
// RUN: %kitxx -### -O1 -ftapir=serial %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
// RUN: %kitxx -### -O2 -ftapir=serial %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE
// RUN: %kitxx -### -O3 -ftapir=serial %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE
// RUN: %kitxx -### -O4 -ftapir=serial %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE
// RUN: %kitxx -### -Os -ftapir=serial %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE
// RUN: %kitxx -### -Oz -ftapir=serial %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NO-STRIPMINE

// Check that the -fstripmine and -fno-stripmine flags override the defaults
// RUN: %kitxx -### -O0 -ftapir=serial -fstripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE
// RUN: %kitxx -### -O1 -ftapir=serial -fstripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE
// RUN: %kitxx -### -O2 -ftapir=serial -fno-stripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
// RUN: %kitxx -### -O3 -ftapir=serial -fno-stripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
// RUN: %kitxx -### -O4 -ftapir=serial -fno-stripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
// RUN: %kitxx -### -Os -ftapir=serial -fno-stripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NO-STRIPMINE
// RUN: %kitxx -### -Oz -ftapir=serial -fstripmine %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE

// STRIPMINE: -fstripmine
// NO-STRIPMINE-NOT: -fstripmine

// Check that the stripmine pass is enabled/disabled correctly
// RUN: %kitxx -mllvm -print-pipeline-passes -O2 -fstripmine -ftapir=serial \
// RUN:      -S -emit-llvm %s | FileCheck %s -check-prefix STRIPMINE-PASS
// RUN: %kitxx -mllvm -print-pipeline-passes -O2 -fno-stripmine -ftapir=serial \
// RUN:      -S -emit-llvm %s | FileCheck %s -check-prefix NO-STRIPMINE-PASS

// STRIPMINE-PASS: loop-stripmine
// NO-STRIPMINE-PASS-NOT: loop-stripmine
