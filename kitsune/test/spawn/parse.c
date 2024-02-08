// RUN: %kitcc -Xclang -verify -fsyntax-only -ftapir=none %s

#include <kitsune.h>

void f() {
  spawn {}
  // expected-error@-1 {{expected identifier}}

  sync;
  // expected-error@-1 {{expected identifier}}
}
