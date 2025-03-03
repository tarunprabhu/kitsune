// ----------------------------------------------------------------------------
// RUN: not %kitxx -### -ftapir=serial -flto -O2 -fuse-ld=lld %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NOT-ALLOWED
//
// RUN: not %kitxx -### -ftapir=serial -flto -O2 --ld-path=something %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix NOT-ALLOWED
//
// NOT-ALLOWED: error: '{{.+}}' cannot be used with -flto in Kitsune
//
// ----------------------------------------------------------------------------
//
// RUN: %kitxx -### -ftapir=serial -flto -O2 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix LINKER-ARGS
//
// LINKER-ARGS: /ld{{(64)?}}.lld
// LINKER-ARGS-SAME: -dynamic-linker
