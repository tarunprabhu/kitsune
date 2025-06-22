// RUN: %kitxx -Xclang -verify -fsyntax-only %s

void mobile_free() {
  int *ptr = nullptr;

  kitsune_mobile_free(ptr);
  // expected-error@-1 {{argument of kitsune_mobile_free must have mobile qualifier}}
  kitsune_mobile_free(*ptr);
  // expected-error@-1 {{argument of kitsune_mobile_free must have mobile qualifier}}
}

void cast_unsafe() {
  int n = 12;
  void *[[kitsune::mobile]] ptr = nullptr;

  __kitsune_mobile_cast_unsafe(ptr);
  // expected-error@-1 {{argument of __kitsune_mobile_cast_arg must be a non-mobile pointer}}
  __kitsune_mobile_cast_unsafe(n);
  // expected-error@-1 {{argument of __kitsune_mobile_cast_arg must be a non-mobile pointer}}
}
