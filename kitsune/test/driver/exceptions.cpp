// If the Kitsune frontend driver is used with a tapir target, exceptions are
// turned off automatically unless explicitly requested by the user. If a tapir
// target is not used, exceptions are enabled as usual.

// RUN: %kitxx -### -ftapir=serial %s 2>&1 \
// RUN:     | FileCheck --check-prefixes=NO-EXCEPTIONS %s
// RUN: %kitxx -### -ftapir=serial -fexceptions %s 2>&1 \
// RUN:     | FileCheck --check-prefixes=EXCEPTIONS %s
// RUN: %kitxx -### -ftapir=serial -fcxx-exceptions %s 2>&1 \
// RUN:     | FileCheck --check-prefixes=EXCEPTIONS %s

// Check that the behavior of the clang frontend has not changed.
// RUN: %clang -### %s 2>&1 | FileCheck --check-prefixes=EXCEPTIONS %s
// RUN: %kitxx -### %s 2>&1 | FileCheck --check-prefixes=EXCEPTIONS %s

// NO-EXCEPTIONS-NOT: -fcxx-exceptions
// NO-EXCEPTIONS-NOT: -fexceptions

// EXCEPTIONS-DAG: -fcxx-exceptions
// EXCEPTIONS-DAG: -fexceptions
