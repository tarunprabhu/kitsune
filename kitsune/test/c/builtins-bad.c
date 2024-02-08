// RUN: %kitcc -Xclang -verify -fsyntax-only -std=c23 %s

#include <stdlib.h>

void mobile_free() {
  int *ptr = NULL;

  kitsune_mobile_free(ptr);
  // expected-error@-1 {{argument of kitsune_mobile_free must have mobile qualifier}}
  kitsune_mobile_free(*ptr);
  // expected-error@-1 {{argument of kitsune_mobile_free must have mobile qualifier}}
}

void cast_unsafe() {
  int n = 12;
  void *[[kitsune::mobile]] ptr = NULL;

  __kitsune_mobile_cast_unsafe(ptr);
  // expected-error@-1 {{argument of __kitsune_mobile_cast_arg must be a non-mobile pointer}}
  __kitsune_mobile_cast_unsafe(n);
  // expected-error@-1 {{argument of __kitsune_mobile_cast_arg must be a non-mobile pointer}}
}
