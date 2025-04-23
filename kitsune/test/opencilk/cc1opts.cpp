// Check that the options required by cc1 make it to the tapir targets.
//
// RUN: %kitxx -cc1 --tapir=opencilk --tapir-verbose -O2 %s -o /dev/null \
// RUN:     -disable-free -emit-llvm \
// RUN:     --tapir-opencilk-runtime-bc="%S/input/libopencilk-abi.bc" 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: 'opencilk' tapir target options
// CHECK: Bitcode file: {{.+}}.bc

// -cc1 needs the correct -I option to the kitsune.h header file. Instead, just
// inline the relevant contents. At some point, we really should handle forall
// differently
#define forall _kitsune_forall

// We need a forall loop so HipABI is entered.
void f(int *c, int n) {
  forall(int i = 0; i < n; ++i) { c[i] = n; }
}
