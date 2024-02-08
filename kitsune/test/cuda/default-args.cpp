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
// CHECK-SAME: -lcudart_static
// CHECK-SAME: -lcuda
//
// ----------------------------------------------------------------------------
// Check that the stripmine pass is disabled by default.
//
// RUN: %kitxx -mllvm -print-pipeline-passes -O2 -ftapir=cuda \
// RUN:     --tapir-cuda-arch=sm_72 -S -emit-llvm %s \
// RUN:     | FileCheck %s -check-prefix STRIPMINE-PASS
//
// STRIPMINE-PASS-NOT: loop-stripmine
//
// -----------------------------------------------------------------------------
// Check that an error is emitted if any of the options required by cc1 are not
// provided.
//
// RUN: not %kitxx -cc1 --tapir=cuda %s -o /dev/null \
// RUN:     --tapir-cuda-virt-arch=compute_72 \
// RUN:     --tapir-cuda-features="+ptx72" \
// RUN:     --tapir-cuda-runtime-bc="%S/input/nvptx.bc" 2>&1 \
// RUN:     | FileCheck %s -check-prefix=MISSING_ARCH
//
// RUN: not %kitxx -cc1 --tapir=cuda %s -o /dev/null \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-cuda-features="+ptx72" \
// RUN:     --tapir-cuda-runtime-bc="%S/input/nvptx.bc" 2>&1 \
// RUN:     | FileCheck %s -check-prefix=MISSING_VIRTARCH
//
// RUN: not %kitxx -cc1 --tapir=cuda %s -o /dev/null \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-cuda-virt-arch=compute_72 \
// RUN:     --tapir-cuda-runtime-bc="%S/input/nvptx.bc" 2>&1 \
// RUN:     | FileCheck %s -check-prefix=MISSING_FEATURES
//
// RUN: not %kitxx -cc1 --tapir=cuda %s -o /dev/null \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-cuda-virt-arch=compute_72 \
// RUN:     --tapir-cuda-features="+ptx72" 2>&1 \
// RUN:     | FileCheck %s -check-prefix=MISSING_RUNTIME_BC
//
// MISSING_ARCH: missing required option '--tapir-cuda-arch='
// MISSING_VIRTARCH: missing required option '--tapir-cuda-virt-arch='
// MISSING_FEATURES: missing required option '--tapir-cuda-features='
// MISSING_RUNTIME_BC: missing required option '--tapir-cuda-runtime-bc='
//
// -----------------------------------------------------------------------------
