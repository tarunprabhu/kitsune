// ----------------------------------------------------------------------------
// The --tapir option requires optimizations, unless the tapir target is nolo.
//
// RUN: not %kitcc --tapir=serial %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --check-prefix=MINOPT
//
// RUN: not %kitcc --tapir=serial -O0 %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --check-prefix=MINOPT
//
// -----------------------------------------------------------------------------
//
// RUN: %kitcc --tapir=nolo -O0 %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --allow-empty --check-prefix=OK
//
// ----------------------------------------------------------------------------
// Sanity check that we don't require optimizations if --tapir is not provided.
//
// RUN: %kitcc %s -c -emit-llvm -o /dev/null \
// RUN:     | FileCheck %s --allow-empty -check-prefix OK
//
// RUN: %kitcc -O0 %s -c -emit-llvm -o /dev/null \
// RUN:     | FileCheck %s --allow-empty -check-prefix OK
//
// -----------------------------------------------------------------------------
// If -flto is given, at least O2 is required. There is true even if the tapir
// target is set to nolo.
//
// RUN: not %kitcc -flto --tapir=serial -O0 %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --check-prefix=MINOPT
//
// RUN: %kitcc -flto --tapir=serial -O1 %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --allow-empty --check-prefix=OK
//
// RUN: %kitcc -flto --tapir=serial -O2 %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --allow-empty --check-prefix=OK
//
// RUN: %kitcc -flto --tapir=serial -O3 %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --allow-empty --check-prefix=OK
//
// RUN: not %kitcc -flto --tapir=serial -Os %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --allow-empty --check-prefix=ERROR
//
// RUN: not %kitcc -flto --tapir=serial -Oz %s -c -emit-llvm -o /dev/null 2>&1 \
// RUN:      | FileCheck %s --allow-empty --check-prefix=ERROR
//
// -----------------------------------------------------------------------------
//
// MINOPT: error: --tapir requires optimization level O1 or higher
// ERROR: unsupported optimization level '-O{{[sz]}}'
// OK-NOT: {{.+}}
