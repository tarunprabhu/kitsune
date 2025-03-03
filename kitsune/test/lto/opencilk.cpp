// REQUIRES: kitsune-opencilk
//
// RUN: %kitxx -### -ftapir=opencilk -O2 -flto %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=ALL
//
// RUN: %kitxx -### -ftapir=serial -O2 -flto %s \
// RUN:     --tapir-verbose 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=TAPIR-VERBOSE
//
// RUN: %kitxx -### -ftapir=serial -O2 -flto %s \
// RUN:     --kitrt-verbose 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=KITRT-VERBOSE

// ALL: /ld{{(64)?}}.lld"
// ALL-SAME: --tapir=opencilk
// ALL-SAME: --tapir-opencilk-abi-bc={{.+}}/libopencilk-abi.bc
// TAPIR-VERBOSE: --tapir-verbose
// KITRT-VERBOSE: --kitrt-verbose
