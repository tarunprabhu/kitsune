// -x hip is not supported in combination with a Tapir target. Check that we
// don't disallow other uses.
//
// RUN: %clang -x hip -nogpuinc -nogpulib --offload-arch=gfx90c \
// RUN:     -fsyntax-only -o /dev/null %s
// RUN: %kitxx -x hip -nogpuinc -nogpulib --offload-arch=gfx90c \
// RUN:     -fsyntax-only -o /dev/null %s
// RUN: not %kitxx -x hip -nogpuinc -nogpulib -o /dev/null %s \
// RUN:     -fsyntax-only --tapir=hip --tapir-hip-arch=gfx90c -O1 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: kitsune does not support the Hip language
