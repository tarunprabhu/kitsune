// Check that a 3D Kokkos::MDRangePolicy produces a 3D loop nest. As of the
// time of writing, this produces a nest where the innermost loop is a tapir
// loop while the others are regular loops. Obviously if we decide to handle
// this differently, this will have to change.
//
// RUN: %kitxx -fkokkos -fkokkos-no-init --tapir=none -S -emit-llvm %s  \
// RUN:     -O1 -fno-stripmine -fno-vectorize -o - %sysroot \
// RUN:     | FileCheck %s

#include <Kokkos_Core.hpp>

extern "C" void f(float *a, int D1, int D2, int D3) {
  Kokkos::parallel_for(
      "initialize_array",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {D1, D2, D3}),
      KOKKOS_LAMBDA(int i, int j, int k) { a[i + j + k] = 0.0; });
}

// CHECK-LABEL: void @f
// CHECK-SAME: %[[D1:[^ ]+]],
// CHECK-SAME: %[[D2:[^ ]+]],
// CHECK-SAME: %[[D3:[^ ]+]])
// CHECK-NEXT: [[ENTRY:.+]]:
// CHECK-NEXT: %[[SYNCREG:.+]] = {{.+}}call token @llvm.syncregion.start
// CHECK: [[LOOPPH_I:^.+]]:
// CHECK: [[LOOP_I:^.+]]:
// CHECK-NEXT: %[[I:.+]] = phi i32 {{.*}}[ %[[NEXT_I:[^ ]+]], %[[LATCH_I:[^ ]+]] ]
// CHECK: [[LOOP_J:^.+]]:
// CHECK-NEXT: %[[J:.+]] = phi i32 {{.*}}[ %[[NEXT_J:[^ ]+]], %[[LATCH_J:[^ ]+]] ]
// CHECK: [[LOOPPH_K:^.+]]:
// CHECK: [[LOOP_K:^.+]]:
// CHECK-NEXT: %[[K:.+]] = phi i32 {{.*}}[ %[[NEXT_K:[^ ]+]], %[[LATCH_K:[^ ]+]] ]
// CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY:[^ ]+]], label %[[LATCH_K]]
// CHECK: [[BODY]]:
// CHECK: reattach within %[[SYNCREG]], label %[[LATCH_K]]
// CHECK: [[LATCH_K]]:
// CHECK-NEXT: %[[NEXT_K]] = add {{.+}} %[[K]], 1
// CHECK-NEXT: %[[COND_K:.+]] = icmp eq i32 %[[NEXT_K]], %[[D3]]
// CHECK-NEXT: br i1 %[[COND_K]], label %[[LATCH_J]], label %[[LOOP_K]]
// CHECK: [[LATCH_J]]:
// CHECK: %[[NEXT_J]] = add {{.+}} %[[J]], 1
// CHECK: br i1 %{{.+}}, label %[[LATCH_I]], label %[[LOOP_J]]
// CHECK: [[LATCH_I]]:
// CHECK: %[[NEXT_I]] = add {{.+}} %[[I]], 1
// CHECK: br i1 %{{.+}}, label %[[SYNC:[^ ]+]], label %[[LOOP_I]]
// CHECK: [[SYNC]]:
// CHECK: sync within %[[SYNCREG]]
