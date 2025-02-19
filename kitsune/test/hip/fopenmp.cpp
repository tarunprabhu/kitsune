// OpenMP offload cannot be used together with a Kitsune tapir target.
//
// RUN: not %kitxx -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -ftapir=hip \
// RUN:     --cuda-gpu-arch=gfx90a -c -O2 %s 2>&1 \
// RUN:     | FileCheck %s

// Running the kitsune frontend without -ftapir is ok
//
// RUN: %kitxx -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:     --cuda-gpu-arch=gfx90a \
// RUN:     %s -c -O2

// CHECK: cannot use OpenMP offload with a tapir target
