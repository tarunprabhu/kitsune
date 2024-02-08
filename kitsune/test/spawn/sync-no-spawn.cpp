// A sync without a corresponding spawn should be an error since the label will
// not have been declared, but this is not currently checked.
// XFAIL: *
// RUN: %kitcc -Xclang -verify -fsyntax-only -ftapir=none %s

#include <kitsune.h>

void f1() {
  sync s;
  // expected-error@10 {{Undeclared label 's' in sync}}
}

void f2() {
  spawn s1 {}
  sync s2;
  // expected-error@15 {{Undeclared label 's2' in sync}}
}
