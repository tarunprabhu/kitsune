// -----------------------------------------------------------------------------
// OpenMP offload cannot be used together with a Kitsune tapir target.
//
// RUN: not %kitxx -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:     --offload-arch=gfx90a -nogpulib -c -O2 %s \
// RUN:     --tapir=hip --tapir-hip-arch=gfx90a  2>&1 \
// RUN:     | FileCheck %s -check-prefix=TAPIR
//
// TAPIR: cannot use OpenMP offload with a tapir target
//
// ----------------------------------------------------------------------------
// Running the kitsune frontend without --tapir is allowed.
//
// RUN: %kitxx -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:     --offload-arch=gfx90a -nogpulib -c -O2 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=NOTAPIR
//
// NOTAPIR-NOT: cannot use OpenMP offload with a tapir target
