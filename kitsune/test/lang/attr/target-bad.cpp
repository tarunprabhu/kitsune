// RUN: %kitxx -Xclang -verify -fsyntax-only --tapir=nolo %s %sysroot

#include <kitsune.h>

int main(int argc, char *argv[]) {
  // expected-error@+1 {{'tapir::target' attribute: unknown value}}
  [[tapir::target("i860")]]
  forall(int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{'tapir::target' attribute requires a string}}
  [[tapir::target(cuda)]]
  forall(int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{'tapir::target' attribute takes one argument}}
  [[tapir::target()]]
  forall(int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{'tapir::target' attribute takes one argument}}
  [[tapir::target("serial","-03")]]
  forall(int i = 0; i < 1024; ++i) {}

  // expected-error@+1 {{'tapir::target' attribute: unsupported statement}}
  [[tapir::target("serial")]]
  if (argc == 1) {
    forall(int i = 0; i < 1024; ++i) {}
  }

  return 0;
}
