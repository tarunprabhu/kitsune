// REQUIRES: kitsune-hip
//
// RUN: %kitxx -### -ftapir=hip -O2 -flto %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=ALL
//
// RUN: %kitxx -### -ftapir=hip -O2 -flto %s \
// RUN:     --tapir-verbose 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=ALL,TAPIR-VERBOSE
//
// RUN: %kitxx -### -ftapir=hip -O2 -flto %s \
// RUN:     --kitrt-verbose 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=ALL,KITRT-VERBOSE
//
// RUN: %kitxx -### -ftapir=hip -O2 -flto %s \
// RUN:     --tapir-hip-arch=gfx906 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=ALL,HIP_ARCH
//
// RUN: %kitxx -### -ftapir=hip -O2 -flto %s \
// RUN:     --tapir-threads-per-block=64 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=ALL,TPB
//
// RUN: %kitxx -### -ftapir=hip -O2 -flto %s \
// RUN:     --tapir-max-threads-per-block=128 2>&1 \
// RUN:     | FileCheck %s -check-prefixes=ALL,MTPB

// ALL: /ld{{(64)?}}.lld"
// TAPIR-VERBOSE: --tapir-verbose
// KITRT-VERBOSE: --kitrt-verbose
// ALL-SAME: --tapir=hip
// HIP_ARCH: --tapir-hip-arch=gfx906
// TPB: --tapir-threads-per-block=64
// MTPB: --tapir-max-threads-per-block=128
