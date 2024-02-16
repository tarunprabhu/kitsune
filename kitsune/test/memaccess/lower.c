// Lowering is not correctly implemented for local variables. There is no
// guarantee that we will end up using metadata to do so, but we might as well
// assume that in the absence of any better alternative.
//
// XFAIL: *
// RUN: %kitcc -ftapir=serial -S -emit-llvm -o - %s | FileCheck %s

int _readonly g;

int f4(int _readwrite *ptr);

void f3(int _writeonly *out) {
  int _readwrite in = g;
  *out = in;
  f4(&in);
}

// CHECK-DAG: declare {{.*}} f4({{.*}}"kitsune.readwrite"{{.*}})
// CHECK-DAG: @g = {{.*}} "kitsune.readonly" i32
// CHECK-DAG: define {{.*}} @f3(ptr {{.*}}"kitsune.writeonly"{{.*}} %out) {
// CHECK: %in = alloca i32, {{.*}} !"kitsune.readwrite"
