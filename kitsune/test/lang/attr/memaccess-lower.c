// Check that the memory access attributes on the source-level entities are
// correctly lowered to LLVM attributes in all supported cases.
//
// TODO: We would like to support lowering on function-local variables as well
// but this has not currently been implemented. When that changes, these tests
// should be updated.
//
// RUN: %kitcc --tapir=nolo -S -emit-llvm -o - %s %sysroot | FileCheck %s

#define NULL ((void*)0)

// CHECK-DAG: @gdecl = {{.*}} #[[GDECL_ATTRS:[0-9]+]]
extern int _writeonly gdecl;

// CHECK-DAG: @gext = {{.*}} #[[GEXT_ATTRS:[0-9]+]]
int _readwrite gext;

// CHECK-DAG: @gint = {{.*}} #[[GINT_ATTRS:[0-9]+]]
static int _readonly gint;

// CHECK-DAG: declare {{.*}} @fdecl(
// CHECK-DAG-SAME: ptr {{.*}}"kitsune.readwrite"{{.*}},
// CHECK-DAG-SAME: ptr {{.*}}"kitsune.readonly"{{.*}},
// CHECK-DAG-SAME: ptr {{.*}}"kitsune.writeonly"{{.*}})
int fdecl(int _readwrite *a1, int _readonly *a2, int _writeonly *a3);

// CHECK-DAG: define {{.*}} @fdef(ptr {{.*}}"kitsune.writeonly"{{.*}} %out)
void fdef(int _writeonly *out) {
  int in = gint + gdecl;
  *out = in;
  gext += fdecl(&in, NULL, NULL);
}

// CHECK-DAG: attributes #[[GDECL_ATTRS]] = { {{.*}}"kitsune.writeonly"{{.*}} }
// CHECK-DAG: attributes #[[GEXT_ATTRS]] = { {{.*}}"kitsune.readwrite"{{.*}} }
// CHECK-DAG: attributes #[[GINT_ATTRS]] = { {{.*}}"kitsune.readonly"{{.*}} }
