// Kitsune does not support .hip files
//
// -----------------------------------------------------------------------------
// Ensure that we still allow clang to compile hip code
//
// RUN: %clang -fsyntax-only -nogpulib -nogpuinc -o /dev/null %s 2>&1 \
// RUN:     | FileCheck --allow-empty --check-prefix=OK %s
//
// -----------------------------------------------------------------------------
// Compiling hip with the kitsune frontend is also fine as long as we don't
// provide a --tapir option
//
// RUN: %kitxx -fsyntax-only -c -nogpulib -nogpuinc -o /dev/null %s \
// RUN:     | FileCheck --allow-empty --check-prefix=OK %s
//
// -----------------------------------------------------------------------------
// We cannot compile hip code with a tapir target
//
// RUN: not %kitxx -fsyntax-only -ftapir=hip -nogpulib -nogpuinc %s 2>&1 \
// RUN:     | FileCheck --check-prefix=BAD %s
//
// -----------------------------------------------------------------------------
//
// OK-NOT: kitsune does not support the Hip language
// BAD: kitsune does not support the Hip language
