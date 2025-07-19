// -----------------------------------------------------------------------------
// Check that the various spellings of Kitsune's device attribute are handled
// correctly. We don't care about the attribute on declarations since that
// doesn't help Kitsune's code generation in any way.
//
// RUN: %kitcc --tapir=nolo -std=c23 -O0 -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes C23
// RUN: %kitcc --tapir=nolo -std=c23 -O0 -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes GNU
//
// C23-LABEL: define {{.+}} @fc23
// C23-SAME: #[[ATTR:[0-9]+]]
// C23: #[[ATTR]] = { {{.*}}kit_device{{.*}} }
//
// GNU-LABEL: define {{.+}} @fgnu
// GNU-SAME: #[[ATTR:[0-9]+]]
// GNU: #[[ATTR]] = { {{.*}}kit_device{{.*}} }
//
// -----------------------------------------------------------------------------
// If --tapir is not provided, the device attributes will not be reflected in
// the IR.
//
// RUN: %kitcc -std=c23 -O0 -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s -check-prefix NOTAPIR
//
// NOTAPIR-NOT: kit_device

[[kitsune::device]] void fc23() {}
__attribute__((kitsune_device)) void fgnu() {}
