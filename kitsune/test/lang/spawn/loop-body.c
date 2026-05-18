// The bodies of loop constructs cannot be spawn statements. It is a hard error
// in the case of forall loops, but undefined behavior in others. Note that this
// is not the same as spawn statements *inside* loop bodies. Here, the spawn
// statement *is* the loop body.
//
// FIXME: Perhaps these should be errors for all loops, not just forall.
//
// FIXME: We should check for spawn statements inside loop bodies - though we
//        haven't decided whether or not we want to support this.
//
// RUN: %kitcc -Xclang -verify --tapir=nolo -fsyntax-only %s %sysroot

#include <kitsune.h>

int main() {
  // expected-warning@+1 {{for loop with spawn statement body has undefined behavior}}
  for (int i = 0; i < 10; i++)
    spawn lbf { }
  sync lbf;

  int i = 0;
  // expected-warning@+1 {{while loop with spawn statement body has undefined behavior}}
  while (i++ < 10)
    spawn lbw { }
  sync lbw;

  int j = 0;
  // expected-warning@+1 {{do loop with spawn statement body has undefined behavior}}
  do
    spawn lbd { }
  while(++j < 10);
  sync lbd;

  // expected-error@+1 {{spawn statements are not allowed in forall loops}}
  forall (int i = 0; i < 10; i++)
    spawn lbfa { }
  sync lbfa;

  return 0;
}
