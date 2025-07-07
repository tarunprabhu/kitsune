// Check that kokkos::parallel_for calls are lowered correctly.
//
// RUN: %kitxx --kokkos --tapir=nolo -S -emit-llvm %sysroot -o - %s 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
// CHECK: detach within %[[SYNCREG:.+]], label %[[BODY:.+]], label %[[INC:.+]]
// CHECK: [[BODY]]
// CHECK store
// CHECK: br label %[[REATTACH:.+]]
// CHECK: [[REATTACH]]:
// CHECK: reattach within %[[SYNCREG]], label %[[INC]]
// CHECK: [[INC]]:
// CHECK: br label {{.+}}, !llvm.loop
// CHECK: [[SYNC:.+]]:
// CHECK: sync within %[[SYNCREG]]

#include "Kokkos_Core.hpp"

extern "C" void f(size_t n) {
  // clang-format off
  Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
  });
  // clang-format on
}
