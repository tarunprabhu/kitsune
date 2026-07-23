// RUN: %kitxx -Xclang -verify -fsyntax-only --tapir=nolo %s %sysroot

#include <kitsune.h>

void f(int n) {
  // expected-error@+1 {{attribute takes one argument}}
  [[kitsune::name()]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{expected string literal as argument of 'name' attribute}}
  [[kitsune::name(cuda)]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{attribute takes one argument}}
  [[kitsune::name("Alfred", "Neuman")]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{attribute only applies to 'forall' statement}}
  [[kitsune::name("this")]]
  spawn a {}

  // expected-error@+1 {{attribute only applies to 'forall' statement}}
  [[kitsune::name("that")]]
  sync a;

  // expected-error@+1 {{attribute only applies to 'forall' statement}}
  [[kitsune::name("other")]]
  while (n) {
    --n;
  }

  // expected-error@+2 {{cannot appear more than once on a statement}}
  // expected-note@+1 {{conflicting attribute is here}}
  [[kitsune::name("mahi"), kitsune::name("mahi")]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{name cannot be an empty string}}
  [[kitsune::name("")]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{name cannot contain spaces}}
  [[kitsune::name("Alfred Neuman")]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{name must contain only printable characters}}
  [[kitsune::name("\bANSI")]]
  forall (int i = 0; i < 1024; ++i) {}
}
