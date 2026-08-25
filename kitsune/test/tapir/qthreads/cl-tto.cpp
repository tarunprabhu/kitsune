// Check that the options specific to the qthreads tapir target make it to the
// tapir target options.
//
// NOTE: Currently, there are no such options, so this is mostly just a
// placeholder and is around for consistency with the tests for the other tapir
// targets.
//
// RUN: %kitxx --tapir=qthreads -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     -mllvm -print-tt-options \
// RUN:     | FileCheck %s -check-prefixes ALL
//
// ALL:          Tapir target options
// ALL:          Primary: qthreads
