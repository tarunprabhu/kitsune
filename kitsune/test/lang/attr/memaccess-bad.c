// RUN: %kitcc --tapir=nolo -Xclang -verify -fsyntax-only %s

// expected-error@+1 {{multiple access qualifiers}}
int f1(_readonly _readwrite int *a, int n) { return a[n]; }

// expected-warning@+1 {{duplicate '_readonly' declaration specifier}}
int f2(_readonly _readonly int *a, int n) { return a[n]; }

// expected-warning@+1 {{duplicate '_readwrite' declaration specifier}}
int f3(_readwrite _readwrite int *a, int n) { return a[n]; }

// expected-warning@+1 {{duplicate '_writeonly' declaration specifier}}
int f4(_writeonly _writeonly int *a, int n) { return a[n]; }

// expected-error@+1 {{multiple access qualifiers}}
int _writeonly _readonly gerr;

// expected-warning@+1 {{duplicate '_readonly' declaration specifier}}
int _readonly _readonly gro;

// expected-warning@+1 {{duplicate '_readwrite' declaration specifier}}
int _readwrite _readwrite grw;

// expected-warning@+1 {{duplicate '_writeonly' declaration specifier}}
int _writeonly _writeonly gwo;
