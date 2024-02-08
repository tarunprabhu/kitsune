// RUN: %kitxx -Xclang -verify -fsyntax-only -ftapir=none %s
// expected-no-diagnostics

#include <kitsune.h>

int main(int argc, char *argv[]) {
  [[tapir::target("none")]]
  forall(int i = 0; i < 1024; ++i) { }

  [[tapir::target("serial")]]
  forall(int i = 0; i < 1024; ++i) { }

  [[tapir::target("cuda")]]
  forall(int i = 0; i < 1024; ++i) { }

  [[tapir::target("hip")]]
  forall(int i = 0; i < 1024; ++i) { }

  [[tapir::target("opencilk")]]
  forall(int i = 0; i < 1024; ++i) { }

  [[tapir::target("openmp")]]
  forall(int i = 0; i < 1024; ++i) { }

  [[tapir::target("qthreads")]]
  forall(int i = 0; i < 1024; ++i) { }

  [[tapir::target("realm")]]
  forall(int i = 0; i < 1024; ++i) { }

  return 0;
}

