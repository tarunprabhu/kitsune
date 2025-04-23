// Check that invalid values passed to the --tapir-threads-per-block option
// emit an appropriate error.
//
// RUN: not %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_72 -O2 %s \
// RUN:     --tapir-threads-per-block= 2>&1 \
// RUN:     | FileCheck %s -check-prefix MISSING
//
// RUN: not %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_72 -O2 %s \
// RUN:     --tapir-threads-per-block=-1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix RANGE
//
// RUN: not %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_72 -O2 %s \
// RUN:     --tapir-threads-per-block=0 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix RANGE
//
// RUN: not %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_72 -O2 %s \
// RUN:     --tapir-threads-per-block=1025 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix RANGE
//
// RUN: %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_72 -O2 %s \
// RUN:     --tapir-threads-per-block=1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix TPBOK
//
// RUN: %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_72 -O2 %s \
// RUN:     --tapir-threads-per-block=1024 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix TPBOK
//
// RUN: not %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_72 -O2 %s \
// RUN:     --tapir-max-threads-per-block= %s 2>&1  \
// RUN:     | FileCheck %s -check-prefix MISSING
//
// RUN: not %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_72 -O2 %s \
// RUN:     --tapir-max-threads-per-block=-1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix UNDERFLOW
//
// RUN: not %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_72 -O2 %s \
// RUN:     --tapir-max-threads-per-block=0 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix UNDERFLOW
//
// RUN: %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_72 -O2 %s \
// RUN:     --tapir-max-threads-per-block=1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix MTPBOK
//
// RUN: %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_72 -O2 %s \
// RUN:     --tapir-max-threads-per-block=1024 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix MTPBOK
//
// RUN: %kitxx -### --tapir=cuda --tapir-cuda-arch=sm_72 -O2 %s \
// RUN:     --tapir-max-threads-per-block=1025 2>&1 \
// RUN:     | FileCheck %s -check-prefix MTPBOK
//
// MISSING: error: argument to '{{.+}}' is missing
// RANGE: error: value of '{{.+}}' not in range
// UNDERFLOW: error: value of '{{.+}}' must be at least 1
// TPBOK: --tapir-threads-per-block={{[0-9]+}}
// MTPBOK: --tapir-max-threads-per-block={{[0-9]+}}
