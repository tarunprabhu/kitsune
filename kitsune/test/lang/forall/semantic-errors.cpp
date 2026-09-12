// RUN: %kitxx -Xclang -verify -fsyntax-only --tapir=nolo %s %sysroot

#include <kitsune.h>

void loop(int n) {
  // clang-format off
  int i;

  // expected-error@+1 {{forall statement must have an initialization expression}}
  forall(; i < n; i++) { }

  // expected-error@+1 {{forall statement must have a condition expression}}
  forall(int j = 0;; j++) { }

  // expected-error@+1 {{forall statement must have an increment expression}}
  forall(int j = 0; j < n;) { }

  // expected-error@+1 {{initializer in a forall statement must be a variable declaration}}
  forall(i = 0; i < n; i++) { }

  // expected-error@+1 {{initializer in a forall statement must declare exactly one variable}}
  forall(int i = 0, j = 0; i < n; i++, j++) { }

  forall(int i = 0; i < n; i++) {
    if (i == 4) {
      // expected-error@+1 {{'break' statements are not allowed in forall loops}}
      break;
    }
  }

  forall(int i = 0; i < n; i++) {
    if (i == 4) {
      // expected-error@+1 {{'continue' statements are not allowed in forall loops}}
      continue;
    }
  }

  // Nested forall's must be perfectly nested relative to an ancestor forall.
  forall (int i = 0; i < n; ++i) {
    int j;
    // expected-error@+2 {{forall loop not perfectly nested}}
    // expected-note@-3 {{loop has an imperfectly nested child}}
    forall(int j = 0; j < n; ++j) {}
  }

  forall (int i = 0; i < n; ++i) {
    // expected-error@+2 {{forall loop not perfectly nested}}
    // expected-note@-2 {{loop has an imperfectly nested child}}
    forall(int j = 0; j < n; ++j) {}
    int k;
  }

  forall (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      // expected-error@+2 {{forall loop not perfectly nested}}
      // expected-note@-3 {{loop has an imperfectly nested child}}
      forall(int k = 0; k < n; ++k) {}

  forall (int i = 0; i < n; ++i)
    if (n)
      // expected-error@+2 {{forall loop not perfectly nested}}
      // expected-note@-3 {{loop has an imperfectly nested child}}
      forall(int j = 0; j < n; ++j) {}

  forall (int i = 0; i < n; ++i) {
    // expected-error@+2 {{forall loop not perfectly nested}}
    // expected-note@-2 {{loop has an imperfectly nested child}}
    forall (int j = 0; j < n; ++j) {}

    // expected-error@+2 {{forall loop not perfectly nested}}
    // expected-note@-6 {{loop has an imperfectly nested child}}
    forall (int j = 0; j < n; ++j) {}
  }

  // clang-format on
}
