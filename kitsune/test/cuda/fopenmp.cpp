// OpenMP offload cannot be used together with a Kitsune tapir target.
//
// RUN: not %kitxx -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -ftapir=cuda \
// RUN:     --cuda-gpu-arch=sm_80 -c -O2 %s \
// RUN:     --libomptarget-nvptx-bc-path=%S/input/nvptx.bc 2>&1 \
// RUN:     | FileCheck %s

// Running the kitsune frontend without -ftapir is ok
//
// RUN: %kitxx -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:     --cuda-gpu-arch=sm_80 -c -O2 %s \
// RUN:     --libomptarget-nvptx-bc-path=%S/input/nvptx.bc

// CHECK: cannot use OpenMP offload with a tapir target
