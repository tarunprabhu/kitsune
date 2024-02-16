// Check that the correct error is emitted if a valid tapir target is specified
// but said target has not been built.
//
// REQUIRES: kitsune-no-realm
//
// RUN: not %kitxx -fsyntax-only -ftapir=realm %s 2>&1 | FileCheck %s
//
// CHECK: Tapir target 'realm' was not enabled when kitsune was built
