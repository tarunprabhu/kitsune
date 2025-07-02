// ----------------------------------------------------------------------------
// The --tapir option requires optimizations, unless the tapir target is none.
//
// RUN: not %kitcc --tapir=serial %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --check-prefix=O1
//
// RUN: not %kitcc --tapir=serial -O0 %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --check-prefix=O1
//
// -----------------------------------------------------------------------------
//
// RUN: %kitcc --tapir=none -O0 %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --allow-empty --check-prefix=OK
//
// ----------------------------------------------------------------------------
// Sanity check that we don't *always* require optimizations.
//
// RUN: %kitcc %s -c -emit-llvm -o /dev/null \
// RUN:     | FileCheck %s --allow-empty -check-prefix OK
//
// RUN: %kitcc -O0 %s -c -emit-llvm -o /dev/null \
// RUN:     | FileCheck %s --allow-empty -check-prefix OK
//
// -----------------------------------------------------------------------------
// If -flto is given, at least O2 is required. There is true even if the tapir
// target is set to 'none'
//
// RUN: not %kitcc -flto --tapir=serial -O0 %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --check-prefix=O2
//
// RUN: not %kitcc -flto --tapir=serial -O1 %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --check-prefix=O2
//
// RUN: not %kitcc -flto --tapir=none -O1 %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --check-prefix=O2
//
// RUN: %kitcc -flto --tapir=serial -O2 %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --allow-empty --check-prefix=OK
//
// RUN: %kitcc -flto --tapir=serial -O3 %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --allow-empty --check-prefix=OK
//
// RUN: %kitcc -flto --tapir=serial -Os %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --allow-empty --check-prefix=OK
//
// RUN: %kitcc -flto --tapir=serial -Oz %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --allow-empty --check-prefix=OK
//
// -----------------------------------------------------------------------------
//
// O1: error: --tapir requires optimization level O1 or higher
// O2: error: --tapir requires optimization level O2 or higher for LTO
// OK-NOT: {{.+}}
