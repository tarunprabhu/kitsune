// RUN: %kitxx -Xclang -verify -fsyntax-only --tapir=nolo %s %sysroot
//
// This checks for various syntax errors in a forall statement. These are
// essentially the same checks as those for a regular for statement. The forall
// has additional semantic constraints but those are not checked here.

#include <kitsune.h>

void f1() {
  // clang-format on

  // expected-error@+1 {{expected ';' in 'for'}}
  forall(int n = 0 n < 10; n++) { }

  // expected-error@+1 {{expected ';' in 'for'}}
  forall(int n = 0; n < 10 n++) { }

  // expected-error@+1 {{expected ';' in 'for'}}
  forall(int n = 0 n < 10; n++) { }

  // expected-error@+1 {{expected ';' in 'for'}}
  forall(int n = 0; n < 10 n++) { }

  // expected-error@+1 {{expected ';' in 'for'}}
  forall(int n = 0 bool b = n < 10; n++) { }

  // expected-error@+1 {{expected ';' in 'for'}}
  forall(int n = 0; bool b = n < 10 n++) { }

  // expected-error@+1 2{{expected ';' in 'for'}}
  forall(int n = 0 n < 10 n++) { }

  // expected-error@+2 {{expected ';' in 'for'}}
  // expected-error@+1 {{forall statement must have an initialization expression}}
  forall(;) { }

  // clang-format off
}
