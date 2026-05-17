// RUN: %kitxx -Xclang -verify -fsyntax-only %s

// This is simply checking that the return type of kitsune_mobile_alloc is
// not compatible with an unattributed pointer.
void mobile_alloc(int n) {
  // expected-error-re@+1 {{cannot initialize a variable {{.+}} with {{.+}}}}
  void *ptr = kitsune_mobile_alloc(n);
}

void mobile_free() {
  int *ptr = nullptr;

  // expected-error-re@+1 {{argument {{.+}} must have mobile qualifier}}
  kitsune_mobile_free(ptr);

  // expected-error-re@+1 {{argument {{.+}} must have mobile qualifier}}
  kitsune_mobile_free(*ptr);
}

void cast_unsafe() {
  int *[[kitsune::mobile]] ptr = nullptr;

  // expected-error-re@+1 {{argument {{.+}} must be a non-mobile pointer}}
  __kitsune_mobile_cast_unsafe(ptr);

  // expected-error-re@+1 {{argument {{.+}} must be a non-mobile pointer}}
  __kitsune_mobile_cast_unsafe(*ptr);
}
