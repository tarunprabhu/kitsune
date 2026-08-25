// -----------------------------------------------------------------------------
// Check that the options provided to kit++ make it to the tapir target options.
// Right now, there are no command line options specific to the opencilk tapir
// target.
//
// -----------------------------------------------------------------------------
// Check that the options only allowed in -cc1 make it to the tapir target
// options.
//
// RUN: %kitxx -cc1 --tapir=opencilk -O2 -emit-llvm -o /dev/null %s \
// RUN:     -disable-free -mllvm -print-tt-options \
// RUN:     --tapir-opencilk-runtime-bc="%S/input/libopencilk-abi.bc" 2>&1 \
// RUN:     | FileCheck %s --check-prefixes ALL,CC1
//
// -----------------------------------------------------------------------------
// ALL:  Tapir target options
// ALL:  Primary: opencilk
// CC1:  Opencilk bitcode file: {{.+}}/libopencilk-abi.bc
