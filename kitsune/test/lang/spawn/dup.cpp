// FIXME: Currently we don't raise an error if we encounter duplicate spawn
// labels.
// XFAIL: *
// RUN: %kitxx -Xclang -verify -fsyntax-only -ftapir=nolo %s

#include <kitsune.h>

void f() {
  spawn s {}
  spawn s {}
  // expected-error@-1 {{Duplicate spawn label. First declared on {{.*}}:7}}
}
