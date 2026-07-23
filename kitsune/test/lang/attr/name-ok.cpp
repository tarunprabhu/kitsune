// RUN: %kitxx -Xclang -verify -fsyntax-only --tapir=nolo %s %sysroot
// expected-no-diagnostics

#include <kitsune.h>

void f() {
  [[kitsune::name("jiaozi")]]
  forall (int i = 0; i < 1024; ++i) {}
}
