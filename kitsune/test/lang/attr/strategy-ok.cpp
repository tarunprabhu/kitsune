// RUN: %kitxx -Xclang -verify -fsyntax-only --tapir=nolo %s
// expected-no-diagnostics

#include <kitsune.h>

int main(int argc, char *argv[]) {
  [[tapir::strategy("seq")]]
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::strategy("dac")]]
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::strategy("gpu")]]
  forall(int i = 0; i < 1024; ++i) { }

  [[tapir::strategy("basic")]]
  forall(int i = 0; i < 1024; ++i) { }

  return 0;
}
