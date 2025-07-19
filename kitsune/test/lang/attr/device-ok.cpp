// -----------------------------------------------------------------------------
// Check that the various spellings of Kitsune's device attribute are handled
// correctly. We don't care about the attribute on declarations since that
// doesn't help Kitsune's code generation in any way.
//
// RUN: %kitxx --tapir=nolo -O0 -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes CXX11
// RUN: %kitxx --tapir=nolo -O0 -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes GNU
//
// CXX11-LABEL: define {{.+}} @fcxx11
// CXX11-SAME: #[[ATTR:[0-9]+]]
// CXX11: #[[ATTR]] = { {{.*}}kit_device{{.*}} }
//
// GNU-LABEL: define {{.+}} @fgnu
// GNU-SAME: #[[ATTR:[0-9]+]]
// GNU: #[[ATTR]] = { {{.*}}kit_device{{.*}} }
//
// -----------------------------------------------------------------------------
// If --tapir is not provided, the device attributes will not be reflected in
// the IR.
//
// RUN: %kitxx -O0 -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s -check-prefix NOTAPIR
//
// NOTAPIR-NOT: kit_device

extern "C" [[kitsune::device]] void fcxx11() {}
extern "C" __attribute__((kitsune_device)) void fgnu() {}
