// Kitsune does not support .cu files
//
// -----------------------------------------------------------------------------
// Ensure that we still allow clang to compile cuda code
//
// RUN: %clang -fsyntax-only -nocudalib -nocudainc -o /dev/null %s 2>&1 \
// RUN:     | FileCheck --allow-empty --check-prefix=OK %s
//
// -----------------------------------------------------------------------------
// Compiling cuda with the kitsune frontend is also fine as long as we don't
// provide a --tapir option
//
// RUN: %kitxx -fsyntax-only -c -nocudalib -nocudainc -o /dev/null %s \
// RUN:     | FileCheck --allow-empty --check-prefix=OK %s
//
// -----------------------------------------------------------------------------
// We cannot compile cuda code with a tapir target
//
// RUN: not %kitxx -fsyntax-only --tapir=cuda -nocudalib -nocudainc %s 2>&1 \
// RUN:     --tapir-cuda-arch=sm_80 \
// RUN:     | FileCheck --check-prefix=BAD %s
//
// -----------------------------------------------------------------------------
//
// OK-NOT: kitsune does not support the Cuda language
// BAD: kitsune does not support the Cuda language
