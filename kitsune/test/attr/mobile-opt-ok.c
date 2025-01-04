// RUN: %kitcc -std=c23 -ftapir=serial -S -emit-llvm -O1 -o - %s | FileCheck %s

double f1(double* [[kitsune::mobile]] ptr) { return *ptr; }

// CHECK-LABEL: @f1
// CHECK: %[[v:.+]] = load double, ptr addrspace(67) %ptr
// CHECK: ret double %[[v]]

char f2(char* [[kitsune::mobile]] ptr, int i) { return ptr[i]; }

// CHECK-LABEL: @f2
// CHECK: %[[gep:.+]] = getelementptr{{.+}} i8, ptr addrspace(67) %ptr, i64 {{.+}}
// CHECK: %[[v:.+]] = load i8, ptr addrspace(67) %[[gep]], {{.+}}
// CHECK: ret i8 %[[v]]

float *[[kitsune::mobile]] f3(float* [[kitsune::mobile]] ptr, int i) {
  return &ptr[i];
}

// CHECK-LABEL: @f3
// CHECK: %[[gep:.+]] = getelementptr{{.+}} float, ptr addrspace(67) %ptr, i64 {{.+}}
// CHECK: ret ptr addrspace(67) %[[gep]]

// Comparing mobile and non-mobile pointers is not allowed in C++, but it is
// allowed in C.
int f9(float* [[kitsune::mobile]] ptr1, float* ptr2) {
  return ptr1 == ptr2;
}

// CHECK-LABEL: @f9
// CHECK: %[[cst:.+]] = addrspacecast ptr {{.+}} to ptr addrspace(67)
// CHECK: icmp eq ptr addrspace(67) %[[cst]], {{.+}}
