// Check that the --tapir-gpu-prefetch and --tapir-gpu-no-prefetch command line
// options are handled correctly.
//
// RUN: %kitxx -### --tapir=cuda -S -emit-llvm -O2 -o - %s \
// RUN:     --tapir-cuda-arch=sm_72 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,PREFETCH
//
// RUN: %kitxx -### --tapir=cuda -S -emit-llvm -O2 -o - %s \
// RUN:     --tapir-gpu-prefetch \
// RUN:     --tapir-cuda-arch=sm_72 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,PREFETCH
//
// RUN: %kitxx -### --tapir=cuda -S -emit-llvm -O2 -o - %s \
// RUN:     --tapir-gpu-no-prefetch \
// RUN:     --tapir-cuda-arch=sm_72 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,NO-PREFETCH
//
// ALL: -cc1
// PREFETCH: --tapir-gpu-prefetch
// NO-PREFETCH: --tapir-gpu-no-prefetch
