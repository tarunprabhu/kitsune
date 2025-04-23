// Check that clang command line options specific to the hip tapir target make
// their way to HipABI.
//
// RUN: %kitxx --tapir=hip --tapir-verbose          \
// RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,COMPILE
//
// RUN: %kitxx --tapir=hip --tapir-verbose --kitrt-verbose \
// RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,RUNTIME
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-arch=gfx906 \
// RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,ARCH
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-threads-per-block=64 %s \
// RUN:     -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefix TPB
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-max-threads-per-block=64 %s \
// RUN:     -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefix MTPB
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-sramecc=off %s \
// RUN:     -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefix SRAMECC_OFF
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-sramecc=on %s \
// RUN:     -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefix SRAMECC_ON
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-sramecc=any %s \
// RUN:     -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefix SRAMECC_ANY
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-xnack=off %s \
// RUN:     -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefix XNACK_OFF
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-xnack=on %s \
// RUN:     -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefix XNACK_ON
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-xnack=any %s \
// RUN:     -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefix XNACK_ANY
//
// RUN: %kitxx --tapir=hip --tapir-verbose -O2 -S -emit-llvm -o /dev/null \
// RUN:     --tapir-hip-arch=gfx1103 --tapir-hip-wavefront64 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes W_64
//
// RUN: %kitxx --tapir=hip --tapir-verbose -O2 -S -emit-llvm -o /dev/null \
// RUN:     --tapir-hip-arch=gfx1103 --tapir-hip-wavefront32 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes W_32
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-abi-version=5 %s \
// RUN:     -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefix ABI_VER_5
//
// ALL: 'hip' tapir target options
// COMPILE:     Runtime verbose: 1
// RUNTIME:     Runtime verbose: 1
// ARCH:        GPU arch: gfx906
// TPB:         Fixed threads/block: 64
// MTPB:        Max threads/block: 64
// SRAMECC_OFF: SRAMECC: off
// SRAMECC_ON:  SRAMECC: on
// SRAMECC_ANY: SRAMECC: any
// XNACK_OFF:   Xnack: off
// XNACK_ON:    Xnack: on
// XNACK_ANY:   Xnack: any
// W_64:        Bitcode files: [
// W_64:          {{.+}}/oclc_wavefrontsize64_on.bc
// W_32:        Bitcode files: [
// W_32:          {{.+}}/oclc_wavefrontsize64_off.bc
// ABI_VER_5:   Bitcode files: [
// ABI_VER_5:     {{.+}}/oclc_abi_version_500.bc

#include <kitsune.h>

// We need a forall loop so the HipABI is entered.
void f(int *c, int n) {
  forall(int i = 0; i < n; ++i) { c[i] = n; }
}
