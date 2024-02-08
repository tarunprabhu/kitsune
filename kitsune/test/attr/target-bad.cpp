// RUN: %kitxx -Xclang -verify -fsyntax-only -ftapir=none %s

#include <kitsune.h>

int main(int argc, char *argv[]) {
  [[tapir::target("i860")]] // expected-error {{unknown tapir target}}
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::target(cuda)]] // expected-error {{'target' attribute requires a string}}
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::target()]] // expected-error {{'target' attribute takes one argument}}
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::target("cuda","-03")]] // expected-error {{'target' attribute takes one argument}}
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::target("cuda")]] // expected-error {{tapir target attribute on unsupported statement}}
  if (argc == 1) {
    forall(int i = 0; i < 1024; ++i) {}
  }

  return 0;
}

