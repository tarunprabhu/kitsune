// Check that the options required by cc1 make it to the tapir targets.
//
// NOTE: This will fail to compile because the hip runtime bitcode files are
// empty. But we only care that the tapir target options are printed correctly.
//
// RUN: not %kitxx -cc1 --tapir=hip --tapir-verbose -O2 %s -o /dev/null \
// RUN:     -disable-free -emit-llvm \
// RUN:     --tapir-hip-arch=gfx906 \
// RUN:     --tapir-hip-sramecc=off \
// RUN:     --tapir-hip-xnack=on \
// RUN:     --tapir-hip-features="-sramecc:+xnack" \
// RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
// RUN:     --tapir-lld="%S/input/ld.lld" 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: 'hip' tapir target options
// CHECK: GPU arch: gfx906
// CHECK: SRAMECC: off
// CHECK: Xnack: on
// CHECK: Target features: -sramecc:+xnack
// CHECK: Bitcode files: [
// CHECK:   {{.+}}/amd.bc
// CHECK: ]
// CHECK: LLD: {{.+}}/input/ld.lld

// -cc1 needs the correct -I option to the kitsune.h header file. Instead, just
// inline the relevant contents. At some point, we really should handle forall
// differently
#define forall _kitsune_forall

// We need a forall loop so HipABI is entered.
void f(int *c, int n) {
  forall(int i = 0; i < n; ++i) { c[i] = n; }
}
