// -----------------------------------------------------------------------------
// Check that the value of threads per block in the kitsune::launch attribute
// is lowered to the correct metadata.
//
// RUN: %kitxx --tapir=hip --tapir-hip-arch=gfx1103 -O1 %s \
// RUN:     -Xclang -disable-llvm-passes -S -emit-llvm -o - \
// RUN:     | FileCheck %s --check-prefix=ATTR
//
// -----------------------------------------------------------------------------
// The command-line argument will not override the value of the launch
// attribute in the code.
//
// RUN: %kitxx --tapir-gpu-tpb=155 --tapir=hip --tapir-hip-arch=gfx1103 \
// RUN:     -Xclang -disable-llvm-passes -S -emit-llvm -O1 -o - %s \
// RUN:     | FileCheck %s --check-prefix=ATTR
//
// ATTR: !{!"tapir.loop.threads.per.block", i32 57}
//
// -----------------------------------------------------------------------------
// The tapir target should prefer the value in the code as well.
//
// RUN: %kitxx --tapir=hip --tapir-hip-arch=gfx1103 \
// RUN:     -O1 -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefix=LAUNCH
//
// RUN: %kitxx --tapir-gpu-tpb=155 --tapir=hip --tapir-hip-arch=gfx1103 \
// RUN:     -S -emit-llvm -O1 -o - %s \
// RUN:     | FileCheck %s --check-prefix=LAUNCH
//
// LAUNCH: call {{.+}} @llvm.kit.async.launch.kernel
// LAUNCH-SAME: i32 57

#include <kitsune.h>

void f(int *a, int n) {
  // clang-format off
  [[kitsune::launch(57)]]
  forall (int i = 0; i < n; ++i)
    a[i] = i;
  // clang-format on
}
