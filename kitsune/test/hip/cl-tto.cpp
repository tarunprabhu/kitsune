// -----------------------------------------------------------------------------
// Check that the options provided to kit++ make it to the tapir target.
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-threads-per-block=64 %s \
// RUN:     -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,TPB
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-max-threads-per-block=64 %s \
// RUN:     -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,MTPB
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-gpu-prefetch %s \
// RUN:     -O2 -S -emit-llvm -o /dev/null 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,PREFETCH
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-gpu-no-prefetch %s \
// RUN:     -O2 -S -emit-llvm -o /dev/null 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,NO-PREFETCH
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-arch=gfx906 %s \
// RUN:     -O2 -S -emit-llvm -o /dev/null 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,ARCH
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-sramecc=off %s \
// RUN:     -O2 -S -emit-llvm -o /dev/null 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,SRAMECC_OFF
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-sramecc=on %s \
// RUN:     -O2 -S -emit-llvm -o /dev/null 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,SRAMECC_ON
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-sramecc=any %s \
// RUN:     -O2 -S -emit-llvm -o /dev/null 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,SRAMECC_ANY
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-xnack=off %s \
// RUN:     -O2 -S -emit-llvm -o /dev/null 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,XNACK_OFF
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-xnack=on %s \
// RUN:     -O2 -S -emit-llvm -o /dev/null 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,XNACK_ON
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-xnack=any %s \
// RUN:     -O2 -S -emit-llvm -o /dev/null 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,XNACK_ANY
//
// RUN: %kitxx --tapir=hip --tapir-verbose -O2 -S -emit-llvm -o /dev/null \
// RUN:     --tapir-hip-arch=gfx1103 --tapir-hip-wavefront64 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,W_64
//
// RUN: %kitxx --tapir=hip --tapir-verbose -O2 -S -emit-llvm -o /dev/null \
// RUN:     --tapir-hip-arch=gfx1103 --tapir-hip-wavefront32 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,W_32
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-abi-version=5 %s \
// RUN:     -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,ABI_VER_5
//
// -----------------------------------------------------------------------------
// Check that the options only allowed in -cc1 make it to the tapir targets.
//
// RUN: %kitxx -cc1 --tapir=hip --tapir-verbose -O2 %s -o /dev/null \
// RUN:     -disable-free -emit-llvm \
// RUN:     --tapir-hip-arch=gfx906 \
// RUN:     --tapir-hip-sramecc=off \
// RUN:     --tapir-hip-xnack=on \
// RUN:     --tapir-hip-features="-sramecc:+xnack" \
// RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
// RUN:     --tapir-lld="%S/input/ld.lld" 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,CC1
//
// -----------------------------------------------------------------------------
// ALL: 'hip' tapir target options
// TPB:             GPU fixed threads/block: 64
// MTPB:            GPU max threads/block: 64
// PREFETCH:        GPU prefetch: 1
// NO-PREFETCH:     GPU prefetch: 0
// ARCH:            Hip arch: gfx906
// SRAMECC_OFF:     Hip sramecc: off
// SRAMECC_ON:      Hip sramecc: on
// SRAMECC_ANY:     Hip sramecc: any
// XNACK_OFF:       Hip xnack: off
// XNACK_ON:        Hip xnack: on
// XNACK_ANY:       Hip xnack: any
// W_64:            Hip target features:{{.*}}+wavefrontsize64
// W_64:            Hip bitcode files: [
// W_64:              {{.+}}/oclc_wavefrontsize64_on.bc
// W_32:            Hip target features:{{.*}}+wavefrontsize32
// W_32:            Hip bitcode files: [
// W_32:              {{.+}}/oclc_wavefrontsize64_off.bc
// ABI_VER_5:       Hip bitcode files: [
// ABI_VER_5:         {{.+}}/oclc_abi_version_500.bc
// CC1:             Hip target features: -sramecc:+xnack
// CC1:             Hip bitcode files: [
// CC1:               {{.+}}/amd.bc
// CC1:             ]
// CC1:             LLD: {{.+}}/input/ld.lld

// We just need some function to ensure that a tapir target object is created.
void f() {}
