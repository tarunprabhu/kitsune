// Check that the correct error is emitted if a valid tapir target is specified
// but said target has not been built.
//
// REQUIRES: kitsune-no-hip
//
// RUN: not %kitxx -fsyntax-only -ftapir=hip %s 2>&1 | FileCheck %s
//
// CHECK: Tapir target 'hip' was not enabled when kitsune was built
