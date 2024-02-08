// Check that the --tapir-gpu-tpb option is handled correctly.
//
// RUN: not %kitxx -### --tapir=hip --tapir-gpu-tpb= %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix MISSING
//
// RUN: not %kitxx -### --tapir=hip --tapir-gpu-tpb=-1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix RANGE
//
// RUN: not %kitxx -### --tapir=hip --tapir-gpu-tpb=0 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix RANGE
//
// RUN: not %kitxx -### --tapir=hip --tapir-gpu-tpb=1025 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix RANGE
//
// RUN: %kitxx -### --tapir=hip --tapir-gpu-tpb=1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix OK
//
// RUN: %kitxx -### --tapir=hip --tapir-gpu-tpb=1024 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix OK
//
// MISSING: error: argument to '{{.+}}' is missing
// RANGE: error: value of '{{.+}}' not in range
// OK: --tapir-gpu-tpb={{[0-9]+}}
