// RUN: %kitcc -Xclang -verify --tapir=nolo -fsyntax-only %s %sysroot

#include <kitsune.h>

int main() {
  for (int i = 0; i < 10; i++)
    // expected-warning@-1 {{for loop with spawn statement body has undefined behavior}}
    spawn lbf { }
  sync lbf;

  int i = 0;
  while (i++ < 10)
    // expected-warning@-1 {{while loop with spawn statement body has undefined behavior}}
    spawn lbw { }
  sync lbw;

  int j = 0;
  do
    // expected-warning@-1 {{do loop with spawn statement body has undefined behavior}}
    spawn lbd { }
  while(++j < 10);
  sync lbd;

  forall (int i = 0; i < 10; i++)
    // expected-error@-1 {{spawn statements are not allowed in forall loops}}
    spawn lbfa { }
  sync lbfa;

  return 0;
}
