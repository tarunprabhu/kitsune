// -----------------------------------------------------------------------------
// If Kokkos mode is enabled, exceptions are turned off automatically unless
// explicitly requested by the user.
//
// RUN: %kitxx -### --kokkos --tapir=nolo %s 2>&1 \
// RUN:     | FileCheck --check-prefixes=NO-EXCEPTIONS %s
//
// RUN: %kitxx -### --kokkos-no-init --tapir=nolo %s 2>&1 \
// RUN:     | FileCheck --check-prefixes=NO-EXCEPTIONS %s
//
// RUN: %kitxx -### --kokkos -fexceptions --tapir=nolo %s 2>&1 \
// RUN:     | FileCheck --check-prefixes=EXCEPTIONS %s
//
// RUN: %kitxx -### --kokkos-no-init -fexceptions --tapir=nolo %s 2>&1 \
// RUN:     | FileCheck --check-prefixes=EXCEPTIONS %s
//
// RUN: %kitxx -### --kokkos -fcxx-exceptions --tapir=nolo %s 2>&1 \
// RUN:     | FileCheck --check-prefixes=EXCEPTIONS %s
//
// RUN: %kitxx -### --kokkos-no-init -fcxx-exceptions --tapir=nolo %s 2>&1 \
// RUN:     | FileCheck --check-prefixes=EXCEPTIONS %s
//
// -----------------------------------------------------------------------------
// Check that the behavior of the clang frontend has not changed.
//
// RUN: %clang -### %s 2>&1 \
// RUN:     | FileCheck --check-prefixes=EXCEPTIONS %s
//
// -----------------------------------------------------------------------------
// NO-EXCEPTIONS-NOT: -fcxx-exceptions
// NO-EXCEPTIONS-NOT: -fexceptions
//
// EXCEPTIONS-DAG: -fcxx-exceptions
// EXCEPTIONS-DAG: -fexceptions
