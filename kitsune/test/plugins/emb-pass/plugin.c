// REQUIRES: kitsune-examples
//
// Check that the embedded module passes in a pass plugin are registered and
// run as expected.
//
// NOTE: The only defined function that will be printed will be the "kernel
// function" - consisting of the body of the tapir loop. We do not check for the
// precise name because it is not guaranteed to be consistent.
//
// RUN: %if kitsune-cuda %{ \
// RUN:   %kitcc --tapir=cuda --tapir-cuda-arch=sm_86 \
// RUN:       -O1 -S -emit-llvm -o /dev/null %s \
// RUN:       -fpass-plugin=%kit-emb-pass-plugin-demo 2>&1 \
// RUN:       | FileCheck %s \
// RUN: %}
//
// RUN: %if kitsune-hip %{ \
// RUN:   %kitcc --tapir=hip --tapir-hip-arch=gfx90a \
// RUN:       -O1 -S -emit-llvm -o /dev/null %s \
// RUN:       -fpass-plugin=%kit-emb-pass-plugin-demo 2>&1 \
// RUN:       | FileCheck %s \
// RUN: %}
//
// CHECK-DAG: declare external_func
// CHECK-DAG: define {{[^ ]+}}
// CHECK-DAG: declare llvm.kit.gpu.thread.id.x
// CHECK-DAG: declare llvm.kit.gpu.block.id.x
// CHECK-DAG: declare llvm.kit.gpu.block.size.x

#include <kitsune.h>

long external_func(long);

void mset(long *a, long n) {
  forall (long i = 0; i < n; ++i)
    a[i] = external_func(i);
}
