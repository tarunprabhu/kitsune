// RUN: %kitxx -Xclang -verify -fsyntax-only --tapir=nolo %s %sysroot

#include <kitsune.h>

int main(int argc, char *argv[]) {
  // expected-error@+1 {{attribute takes one argument}}
  [[tapir::target()]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{attribute requires a string}}
  [[tapir::target(cuda)]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{attribute takes one argument}}
  [[tapir::target("serial", "-O3")]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{unknown value}}
  [[tapir::target("i860")]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{unsupported statement}}
  [[tapir::target("serial")]]
  spawn a {}

  // expected-error@+1 {{unsupported statement}}
  [[tapir::target("serial")]]
  sync a;

  // expected-error@+1 {{unsupported statement}}
  [[tapir::target("serial")]]
  if (argc == 1) {
    forall (int i = 0; i < 1024; ++i) {}
  }

  // expected-error@+2 {{cannot appear more than once on a statement}}
  // expected-note@+1 {{conflicting attribute is here}}
  [[tapir::target("pthreads"), tapir::target("serial")]]
  forall (int i = 0; i < 1024; ++i) {}

  return 0;
}
