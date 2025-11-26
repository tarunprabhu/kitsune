// Check that the tapir target options specific to the serial tapir target are
// set correctly depending on the corresponding frontend options.
//
// Currently, there are no options specific to the serial tapir target. We just
// check that the tapir target ID is set correctly.
//
// RUN: %kitcc --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     -mllvm -dump-tapir-target-options \
// RUN:     | FileCheck %s -check-prefixes ALL
//
// ALL: Tapir target options
// ALL: Primary: serial
