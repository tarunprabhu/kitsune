// RUN: %kitxx -Xclang -verify -fsyntax-only -O1 %s \
// RUN:     --tapir=cuda --tapir-cuda-arch=sm_86

#include <kitsune.h>

int main(int argc, char *argv[]) {
  // clang-format off

  // expected-error@+1 {{attribute takes one argument}}
  [[kitsune::launch()]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{attribute takes one argument}}
  [[kitsune::launch(32, 64)]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{attribute requires a positive integral compile time constant expression}}
  [[kitsune::launch(1 + 2.3)]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{attribute requires a positive integral compile time constant expression}}
  [[kitsune::launch("32")]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{attribute requires a positive integral compile time constant expression}}
  [[kitsune::launch(2.3)]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{attribute requires a positive integral compile time constant expression}}
  [[kitsune::launch(-1)]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{value not in range [0,1024]}}
  [[kitsune::launch(1025)]]
  forall (int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{attribute only applies to 'forall' statement}}
  [[kitsune::launch(32)]]
  spawn s {}

  // expected-error@+1 {{attribute only applies to 'forall' statement}}
  [[kitsune::launch(32)]]
  sync s;

  // expected-error@+1 {{attribute only applies to 'forall' statement}}
  [[kitsune::launch(45)]]
  if (argc == 1) {
    forall (int i = 0; i < 1024; ++i) {}
  }

  // expected-error@+2 {{cannot appear more than once on a statement}}
  // expected-note@+1 {{conflicting attribute is here}}
  [[kitsune::launch(128), kitsune::launch(256)]]
  forall (int i = 0; i < 1024; ++i) {}

  // clang-format on
  return 0;
}
