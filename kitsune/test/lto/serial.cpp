// RUN: %kitxx -### -ftapir=serial -O2 -flto %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=ALL
//
// RUN: %kitxx -### -ftapir=serial -O2 -flto %s \
// RUN:     --tapir-verbose 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=TAPIR_VERBOSE
//
// RUN: %kitxx -### -ftapir=serial -O2 -flto %s \
// RUN:     --kitrt-verbose 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=KITRT_VERBOSE

// ALL: /ld{{(64)?}}.lld"
// ALL-SAME: --tapir=serial
// TAPIR_VERBOSE: --tapir-verbose
// KITRT_VERBOSE: --kitrt-verbose
