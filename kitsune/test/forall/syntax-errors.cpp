// RUN: %kitxx -Xclang -verify -fsyntax-only -ftapir=none %s
//
// This checks for various syntax errors in a forall statement. These are
// essentially the same checks as those for a regular for statement. The forall
// has additional semantic constraints but those are not checked here.

#include <kitsune.h>

void f1() {
  forall(int n = 0 n < 10; n++) {
    // expected-error@10 {{expected ';' in 'for'}}
  }

  forall(int n = 0; n < 10 n++) {
    // expected-error@14 {{expected ';' in 'for'}}
  }

  forall(int n = 0 n < 10; n++) {
    // expected-error@18 {{expected ';' in 'for'}}
  }

  forall(int n = 0; n < 10 n++) {
    // expected-error@22 {{expected ';' in 'for'}}
  }

  forall(int n = 0 bool b = n < 10; n++) {
    // expected-error@26 {{expected ';' in 'for'}}
  }

  forall(int n = 0; bool b = n < 10 n++) {
    // expected-error@30 {{expected ';' in 'for'}}
  }

  forall(int n = 0 n < 10 n++) {
    // expected-error@34 2{{expected ';' in 'for'}}
  }

  forall(;) {
    // expected-error@38 {{expected ';' in 'for'}}
    // expected-error@38 {{forall statement must have an initialization expression}}
  }
}
