// -----------------------------------------------------------------------------
// If the --tapir option is not provided, the Kitsune passes are not run.
//
// RUN: %kitxx -O0 -mllvm -debug-pass=Structure %s -c -o /dev/null 2>&1 \
// RUN:     | FileCheck -check-prefix DEFAULT %s
//
// RUN: %kitxx -O1 -mllvm -debug-pass=Structure %s -c -o /dev/null 2>&1 \
// RUN:     | FileCheck -check-prefix DEFAULT %s
//
// RUN: %kitxx -O2 -mllvm -debug-pass=Structure %s -c -o /dev/null 2>&1 \
// RUN:     | FileCheck -check-prefix DEFAULT %s
//
// RUN: %kitxx -O3 -mllvm -debug-pass=Structure %s -c -o /dev/null 2>&1 \
// RUN:     | FileCheck -check-prefix DEFAULT %s
//
// RUN: %kitxx -Os -mllvm -debug-pass=Structure %s -c -o /dev/null 2>&1 \
// RUN:     | FileCheck -check-prefix DEFAULT %s
//
// RUN: not %kitxx -Oz -mllvm -debug-pass=Structure %s -c -o /dev/null 2>&1 \
// RUN:     | FileCheck -check-prefix ERROR %s
//
// DEFAULT-NOT: Strip Kitsune address spaces
// DEFAULT-NOT: Lower Kitsune intrinsics
// DEFAULT-NOT: Generate Kitsune fat binaries
//
// ERROR: unsupported optimization level '-Oz'
//
// -----------------------------------------------------------------------------
// If the --tapir option is provided, the Kitsune passes are run at all
// optimization levels.
//
// RUN: %kitxx -O0 --tapir=nolo %s -c -o /dev/null \
// RUN:     -mllvm -debug-pass=Structure 2>&1 \
// RUN:     | FileCheck %s -check-prefix TAPIR
//
// RUN: %kitxx -O1 --tapir=nolo %s -c -o /dev/null \
// RUN:     -mllvm -debug-pass=Structure 2>&1 \
// RUN:     | FileCheck %s -check-prefix TAPIR
//
// RUN: %kitxx -O2 --tapir=nolo %s -c -o /dev/null \
// RUN:     -mllvm -debug-pass=Structure 2>&1 \
// RUN:     | FileCheck %s -check-prefix TAPIR
//
// RUN: %kitxx -O3 --tapir=nolo %s -c -o /dev/null \
// RUN:     -mllvm -debug-pass=Structure 2>&1 \
// RUN:     | FileCheck %s -check-prefix TAPIR
//
// RUN: %kitxx -Os --tapir=nolo %s -c -o /dev/null \
// RUN:     -mllvm -debug-pass=Structure 2>&1 \
// RUN:     | FileCheck %s -check-prefix TAPIR
//
// RUN: not %kitxx -Oz --tapir=nolo %s -c -o /dev/null \
// RUN:     -mllvm -debug-pass=Structure 2>&1 \
// RUN:     | FileCheck %s -check-prefix ERROR
//
// TAPIR: ModulePass Manager
// TAPIR-NEXT: FunctionPass Manager
// TAPIR-NEXT: Lower Kitsune intrinsics
// TAPIR-NEXT: Strip Kitsune address spaces
// TAPIR-NEXT: Generate Kitsune fat binaries
// TAPIR-NEXT: Pre-ISel Intrinsic Lowering
//
// -----------------------------------------------------------------------------
