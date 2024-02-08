// RUN: %kitcc -Xclang -verify -fsyntax-only --tapir=nolo %s %sysroot

#include <kitsune.h>

void f() {
  // expected-error@+1 {{expected identifier}}
  spawn {}

  // expected-error@+1 {{expected identifier}}
  sync;
}
