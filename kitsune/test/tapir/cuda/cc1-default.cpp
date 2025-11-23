// -----------------------------------------------------------------------------
// Check that the default options added to the internal command lines (for -cc1
// and the linker) are as expected.
//
// RUN: %kitxx -### --tapir=cuda -O2 %s 2>&1 | FileCheck %s
//
// CHECK: -cc1
// CHECK-SAME: --tapir=cuda
// CHECK-SAME: --tapir-cuda-arch=sm_{{[0-9]+}}
// CHECK-SAME: --tapir-cuda-virt-arch=compute_{{[0-9]+}}
// CHECK-SAME: --tapir-cuda-features={{[^"]+}}"
// CHECK-SAME: --tapir-cuda-runtime-bc={{[^"]+}}.bc"
// CHECK-SAME: --tapir-gpu-prefetch
//
// Stripmining is disabled by default on GPU tapir targets.
//
// CHECK-NOT: -fstripmine
//
// It is a pain to check for the actual linker executable. There are far too
// many options depending on the platform, so just check the next line for the
// expected linker flags.
//
// CHECK-NEXT: -lkitrt
// CHECK-SAME: -lcuda
// CHECK-SAME: -lcudart_static
//
// ----------------------------------------------------------------------------
// Check that the stripmine pass is disabled by default.
//
// RUN: %kitxx -mllvm -print-pipeline-passes -O2 --tapir=cuda \
// RUN:     --tapir-cuda-arch=sm_72 -S -emit-llvm %s \
// RUN:     | FileCheck %s -check-prefix STRIPMINE-PASS
//
// STRIPMINE-PASS-NOT: loop-stripmine
//
// -----------------------------------------------------------------------------
