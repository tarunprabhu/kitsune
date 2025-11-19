// -x cuda is not supported in combination with a Tapir target. Check that we
// don't disallow other uses.
//
// RUN: %clang -x cuda -nocudalib -nocudainc --offload-arch=sm_80 \
// RUN:     -fsyntax-only -o /dev/null %s
// RUN: %kitxx -x cuda -nocudalib -nocudainc --offload-arch=sm_80 \
// RUN:     -fsyntax-only -S -emit-llvm -o /dev/null %s
// RUN: not %kitxx -x cuda -nocudalib -nocudainc -o /dev/null %s \
// RUN:     -fsyntax-only --tapir=cuda --tapir-cuda-arch=sm_80 -O1 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: kitsune does not support the Cuda language
