// ----------------------------------------------------------------------------
// The --tapir option requires optimizations, unless the tapir target is none.
//
// RUN: not %kitcc --tapir=serial %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --check-prefix=ERROR
//
// RUN: not %kitcc --tapir=serial -O0 %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --check-prefix=ERROR
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
//
// ERROR: error: --tapir requires optimization level O1 or higher
// OK-NOT: {{.+}}
