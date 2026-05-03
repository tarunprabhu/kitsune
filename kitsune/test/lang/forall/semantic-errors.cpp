// RUN: %kitxx -Xclang -verify -fsyntax-only --tapir=nolo %s %sysroot

#include <kitsune.h>

void loop() {
  // clang-format off
  int i;

  // expected-error@+1 {{forall statement must have an initialization expression}}
  forall(; i < 10; i++) { }

  // expected-error@+1 {{forall statement must have a condition expression}}
  forall(int j = 0;; j++) { }

  // expected-error@+1 {{forall statement must have an increment expression}}
  forall(int j = 0; j < 10;) { }

  // expected-error@+1 {{initializer in a forall statement must be a variable declaration}}
  forall(i = 0; i < 10; i++) { }

  // expected-error@+1 {{initializer in a forall statement must declare exactly one variable}}
  forall(int i = 0, j = 0; i < 10; i++, j++) { }

  forall(int i = 0; i < 10; i++) {
    if (i == 4) {
      // expected-error@+1 {{forall body may not have a break statement}}
      break;
    }
  }

  // continue statements are allowed in a forall.
  forall(int i = 0; i < 10; i++) {
    if (i == 4) {
      // expected-no-error {{forall body may not have a continue statement}}
      continue;
    }
  }

  // clang-format on
}
