// OpenMP offload cannot be used together with a Kitsune tapir target.
//
// RUN: not %kitxx -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:     --cuda-gpu-arch=gfx90a -ftapir=hip -nogpulib -c -O2 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=TAPIR
//
// Running the kitsune frontend without -ftapir is ok
//
// RUN: %kitxx -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:     --cuda-gpu-arch=gfx90a -nogpulib -c -O2 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=NOTAPIR
///
// TAPIR: cannot use OpenMP offload with a tapir target
// NOTAPIR-NOT: cannot use OpenMP offload with a tapir target
