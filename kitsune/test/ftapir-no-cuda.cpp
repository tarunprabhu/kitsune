// Check that the correct error is emitted if a valid tapir target is specified
// but said target has not been built.
//
// REQUIRES: kitsune-no-cuda
//
// RUN: not %kitxx -fsyntax-only -ftapir=cuda %s 2>&1 | FileCheck %s
//
// CHECK: Tapir target 'cuda' was not enabled when kitsune was built
