// RUN: %kitcc -ftapir=none -Xclang -verify -fsyntax-only %s

int f1(_readonly _readwrite int* a, int n) {
  // expected-error@-1 {{multiple access qualifiers}}
  return a[n];
}

int f2(_readonly _readonly int* a, int n) {
  // expected-warning@-1 {{duplicate '_readonly' declaration specifier}}
  return a[n];
}

int f3(_readwrite _readwrite int* a, int n) {
  // expected-warning@-1 {{duplicate '_readwrite' declaration specifier}}
  return a[n];
}

int f4(_writeonly _writeonly int* a, int n) {
  // expected-warning@-1 {{duplicate '_writeonly' declaration specifier}}
  return a[n];
}

int _writeonly _readonly gerr;
int _readonly _readonly gro;
int _readwrite _readwrite grw;
int _writeonly _writeonly gwo;
// expected-error@-4 {{multiple access qualifiers}}
// expected-warning@-4 {{duplicate '_readonly' declaration specifier}}
// expected-warning@-4 {{duplicate '_readwrite' declaration specifier}}
// expected-warning@-4 {{duplicate '_writeonly' declaration specifier}}
