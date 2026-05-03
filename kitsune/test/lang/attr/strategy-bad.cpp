// RUN: %kitxx -Xclang -verify -fsyntax-only --tapir=nolo %s %sysroot

#include <kitsune.h>

int main(int argc, char *argv[]) {
  // expected-error@+1 {{'tapir::strategy' attribute: unknown value}}
  [[tapir::strategy("greedy")]]
  forall(int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{'tapir::strategy' attribute requires a string}}
  [[tapir::strategy(seq)]]
  forall(int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{'tapir::strategy' attribute takes one argument}}
  [[tapir::strategy()]]
  forall(int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{'tapir::strategy' attribute takes one argument}}
  [[tapir::strategy("seq", "gpu")]]
  forall(int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{'tapir::strategy' attribute only applies to 'forall' statement}}
  [[tapir::strategy("gpu")]]
  spawn s {}

  // expected-error@+1 {{'tapir::strategy' attribute only applies to 'forall' statement}}
  [[tapir::strategy("dac")]]
  if (argc == 1) {
    forall(int i = 0; i < 1024; ++i) {}
  }

  return 0;
}
