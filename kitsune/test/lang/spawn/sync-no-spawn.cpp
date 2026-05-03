// A sync without a corresponding spawn should be an error since the label will
// not have been declared, but this is not currently checked.
// XFAIL: *
// RUN: %kitcc -Xclang -verify -fsyntax-only --tapir=nolo %s

#include <kitsune.h>

void f1() {
  // expected-error@+1 {{Undeclared label 's' in sync}}
  sync s;
}

void f2() {
  spawn s1 {}

  // expected-error@+1 {{Undeclared label 's2' in sync}}
  sync s2;
}
