// RUN: %kitcc -Xclang -verify -fsyntax-only -ftapir=nolo %s

#include <kitsune.h>

void f() {
  spawn {}
  // expected-error@-1 {{expected identifier}}

  sync;
  // expected-error@-1 {{expected identifier}}
}
