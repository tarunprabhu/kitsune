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
// CHECK-NOT: -fstripmine
//
// The next line is expected to be the linker invocation. Since it is difficult
// to reliably check the name of the linker executable, just check for the
// expected linker flags.
//
// CHECK-NEXT: -lkitrt
// CHECK-SAME: -lcuda
// CHECK-SAME: -lcudart_static
//
// ----------------------------------------------------------------------------
// Check that the stripmine pass is disabled by default. This checks that the
// the pipeline tuning options object value is set correctly by default.
//
// RUN: %kitxx -mllvm -print-pipeline-passes -O2 \
// RUN:     --tapir=cuda --tapir-cuda-arch=sm_72 \
// RUN:     -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix STRIPMINE-PASS
//
// STRIPMINE-PASS-NOT: loop-stripmine
//
// -----------------------------------------------------------------------------
