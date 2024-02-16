// RUN: %clang -Xclang -verify -ftapir=none -fsyntax-only %s

#include <kitsune.h>
#include <stdio.h>

int main() {
  for (int i = 0; i < 10; i++)
    // expected-warning@7 {{for loop with spawn statement body has undefined behavior}}
    spawn lbf { printf("Hello %d\n", i); }
  sync lbf;

  int i = 0;
  while (i++ < 10)
    // expected-warning@13 {{while loop with spawn statement body has undefined behavior}}
    spawn lbw { printf("Hello %d\n", i); }
  sync lbw;

  int j = 0;
  do
    // expected-warning@19 {{do loop with spawn statement body has undefined behavior}}
    spawn lbd { printf("Hello %d\n", j); }
  while(++j < 10);
  sync lbd;

  forall (int i = 0; i < 10; i++)
    // expected-error@25 {{spawn statements are not allowed in forall loops}}
    spawn lbfa { printf("Hello %d\n", i); }
  sync lbfa;

  return 0;
}
