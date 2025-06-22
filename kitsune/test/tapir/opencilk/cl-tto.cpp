// -----------------------------------------------------------------------------
// Check that the options provided to kit++ make it to the tapir target.
// Right now, there are no command line options specific to the opencilk tapir
// target.
//
// -----------------------------------------------------------------------------
// Check that the options only allowed in -cc1 make it to the tapir targets.
//
// RUN: %kitxx -cc1 --tapir=opencilk --tapir-verbose -O2 %s -o /dev/null \
// RUN:     -disable-free -emit-llvm \
// RUN:     --tapir-opencilk-runtime-bc="%S/input/libopencilk-abi.bc" 2>&1 \
// RUN:     | FileCheck %s --check-prefixes ALL,CC1
//
// -----------------------------------------------------------------------------
// ALL: 'opencilk' tapir target options
// CC1:       Opencilk bitcode file: {{.+}}/libopencilk-abi.bc

// We just need some function to ensure that a tapir target object is created.
void f() {}
