// Check that the device attribute cannot be applied to global variables. This
// may change in the future, but for now, we do not support it.
//
// RUN: not %kitxx --tapir=nolo -O0 -fsyntax-only %s 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: error: 'device' attribute only applies to functions

extern "C" [[kitsune::device]] int c = 42;
