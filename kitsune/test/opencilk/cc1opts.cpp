// Check that the options required by cc1 make it to the tapir targets.
//
// RUN: %kitxx -cc1 --tapir=opencilk --tapir-verbose -O2 %s -o /dev/null \
// RUN:     -disable-free -emit-llvm \
// RUN:     --tapir-opencilk-runtime-bc="%S/input/libopencilk-abi.bc" 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: 'opencilk' tapir target options
// CHECK: Bitcode file: {{.+}}.bc

// We just need some function to ensure that a tapir target object is created.
void f() {}
