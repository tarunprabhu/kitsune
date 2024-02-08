// Check that the --tapir-gpu-max-tpb option is handled correctly.
//
// RUN: not %kitxx -### --tapir=hip --tapir-gpu-max-tpb= -O1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix MISSING
//
// RUN: not %kitxx -### --tapir=hip --tapir-gpu-max-tpb=-1 -O1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix RANGE
//
// RUN: not %kitxx -### --tapir=hip --tapir-gpu-max-tpb=0 -O1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix RANGE
//
// RUN: not %kitxx -### --tapir=hip --tapir-gpu-max-tpb=1025 -O1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix RANGE
//
// RUN: %kitxx -### --tapir=hip --tapir-gpu-max-tpb=1 -O1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix OK
//
// RUN: %kitxx -### --tapir=hip --tapir-gpu-max-tpb=1024 -O1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix OK
//
// MISSING: error: argument to '{{.+}}' is missing
// RANGE: error: value of '{{.+}}' not in range [1,1024]
// OK: --tapir-gpu-max-tpb={{[0-9]+}}
