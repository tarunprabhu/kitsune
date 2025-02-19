// OpenMP offload cannot be used together with a Kitsune tapir target.
//
// RUN: not %kitxx -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu -ftapir=serial \
// RUN:     %s -c -O2 2>&1 \
// RUN:     | FileCheck %s

// Running the kitsune frontend without -ftapir is ok
//
// RUN: %kitxx -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu %s -c -O2

// CHECK: cannot use OpenMP offload with a tapir target
