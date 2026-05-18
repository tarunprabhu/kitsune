// RUN: %kitcc -std=c23 --tapir=nolo -S -emit-llvm -O0 -o - %s %sysroot \
// RUN:     | FileCheck %s

#define NULL ((void*)0)

typedef struct {
  int n;
} S1;

// CHECK: @s1 ={{.*}} global ptr addrspace(67) null
S1 *[[kitsune::mobile]] s1 = NULL;

// CHECK: void @f1(ptr addrspace(67) {{.+}})
void f1(int *[[kitsune::mobile]] ptr) {}

// CHECK: ptr addrspace(67) @f2(ptr addrspace(67) {{.+}})
void *[[kitsune::mobile]] f2(void *[[kitsune::mobile]] ptr) {
  return ptr;
}

// CHECK-LABEL: @f3
// CHECK: %[[local:.+]] = alloca ptr addrspace(67)
// CHECK: store ptr addrspace(67) null, ptr %[[local]]
void f3() { void *[[kitsune::mobile]] ptr = NULL; }

// Calling a function with a mobile attribute on a parameter must match the
// attributes exactly.
// CHECK-LABEL: @f4
// CHECK: call void @f1(ptr addrspace(67) {{.+}})
void f4(int *[[kitsune::mobile]] ptr) { f1(ptr); }

// Return types must also match
// CHECK-LABEL: @f5
// CHECK: %[[ret:.+]] = call ptr addrspace(67) @f2(ptr addrspace(67) {{.*}}null)
// CHECK: ret ptr addrspace(67) %[[ret]]
void *[[kitsune::mobile]] f5() { return f2(NULL); }

// Passing, or returning a mobile pointer to function which does not accept one
// requires a cast.
// CHECK-LABEL: @f7
// CHECK: %[[cst:.+]] = addrspacecast ptr addrspace(67) {{.+}} to ptr
// CHECK: call {{.*}}i8 @f6(ptr {{.*}}%[[cst]])
char f6(char *ptr) { return *ptr; }
char f7(char *[[kitsune::mobile]] ptr) { return f6((char *)ptr); }

// CHECK-LABEL: @f8
// CHECK: %[[cst:.+]] = addrspacecast ptr addrspace(67) {{.+}} to ptr
// CHECK: ret ptr %[[cst]]
float *f8(char *[[kitsune::mobile]] ptr) { return (float *)ptr; }
