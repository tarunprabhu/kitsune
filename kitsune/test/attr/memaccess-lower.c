// Check that the memory access attributes on the source-level entities are
// correctly lowered to LLVM attributes in all supported cases.
//
// TODO: We would like to support lowering on function-local variables as well
// but this has not currently been implemented. When that changes, these tests
// should be updated.
//
// RUN: %kitcc -ftapir=serial -S -emit-llvm -o - %s | FileCheck %s

#include <stdlib.h>

extern int _writeonly gdecl;

int _readwrite gext;

static int _readonly gint;

int fdecl(int _readwrite *a1, int _readonly *a2, int _writeonly *a3);

void fdef(int _writeonly *out) {
  int in = gint + gdecl;
  *out = in;
  gext += fdecl(&in, NULL, NULL);
}

// CHECK-DAG: @gdecl = {{.*}} #[[GDECL_ATTRS:[0-9]+]]
// CHECK-DAG: @gext = {{.*}} #[[GEXT_ATTRS:[0-9]+]]
// CHECK-DAG: @gint = {{.*}} #[[GINT_ATTRS:[0-9]+]]

// CHECK-DAG: declare {{.*}} @fdecl(
// CHECK-DAG-SAME: ptr {{.*}}"kitsune.readwrite"{{.*}},
// CHECK-DAG-SAME: ptr {{.*}}"kitsune.readonly"{{.*}},
// CHECK-DAG-SAME: ptr {{.*}}"kitsune.writeonly"{{.*}})

// CHECK-DAG: define {{.*}} @fdef(ptr {{.*}}"kitsune.writeonly"{{.*}} %out)

// CHECK-DAG: attributes #[[GDECL_ATTRS]] = { {{.*}}"kitsune.writeonly"{{.*}} }
// CHECK-DAG: attributes #[[GEXT_ATTRS]] = { {{.*}}"kitsune.readwrite"{{.*}} }
// CHECK-DAG: attributes #[[GINT_ATTRS]] = { {{.*}}"kitsune.readonly"{{.*}} }
