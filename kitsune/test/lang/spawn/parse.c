// RUN: %kitcc -Xclang -verify -fsyntax-only --tapir=nolo %s %sysroot

#include <kitsune.h>

void f() {
  spawn {}
  // expected-error@-1 {{expected identifier}}

  sync;
  // expected-error@-1 {{expected identifier}}
}
