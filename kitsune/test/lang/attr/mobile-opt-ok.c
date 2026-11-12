// RUN: %kitcc -std=c23 --tapir=serial -S -emit-llvm -O1 -o - %s \
// RUN:     | FileCheck %s

// CHECK-LABEL: @f1
// CHECK-SAME: %[[PTR1:[^)]+]]
// CHECK: %[[V1:.+]] = load double, ptr addrspace(67) %[[PTR1]]
// CHECK: ret double %[[V1]]
double f1(double* [[kitsune::mobile]] ptr) { return *ptr; }

// CHECK-LABEL: @f2
// CHECK-SAME: %[[PTR2:[^,]+]]
// CHECK: %[[GEP2:.+]] = getelementptr {{.+}}, ptr addrspace(67) %[[PTR2]]
// CHECK: %[[V2:.+]] = load i8, ptr addrspace(67) %[[GEP2]], {{.+}}
// CHECK: ret i8 %[[V2]]
char f2(char* [[kitsune::mobile]] ptr, int i) { return ptr[i]; }

// CHECK-LABEL: @f3
// CHECK-SAME: %[[PTR3:[^,]+]]
// CHECK: %[[GEP3:.+]] = getelementptr {{.+}}, ptr addrspace(67) %[[PTR3]]
// CHECK: ret ptr addrspace(67) %[[GEP3]]
float *[[kitsune::mobile]] f3(float* [[kitsune::mobile]] ptr, int i) {
  return &ptr[i];
}

// Comparing mobile and non-mobile pointers is not allowed in C++, but it is
// allowed in C.
//
// CHECK: @f9(ptr addrspace(67) {{.+}}%[[PTR9:.+]]
// CHECK: %[[CST9:.+]] = addrspacecast ptr {{.+}} to ptr addrspace(67)
// CHECK: icmp eq ptr addrspace(67)
// CHECK-SAME-DAG: %[[CST9]]
// CHECK-SAME-DAG: %[[PTR9]]
int f9(float* [[kitsune::mobile]] ptr1, float* ptr2) {
  return ptr1 == ptr2;
}
