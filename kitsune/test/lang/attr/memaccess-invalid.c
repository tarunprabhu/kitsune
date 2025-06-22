// RUN: %kitcc -ftapir=none -Xclang -verify -fsyntax-only %s

// Attributes are not valid on functions.
int _readonly f5(int* a, int n);
int _readwrite f6(int* a, int n);
int _writeonly f7(int* a, int n);
// expected-error@-3 {{'_readonly' attribute only applies to global variables and parameters}}
// expected-error@-3 {{'_readwrite' attribute only applies to global variables and parameters}}
// expected-error@-3 {{'_writeonly' attribute only applies to global variables and parameters}}

int _readonly f8(int* a, int n) {
  // expected-error@-1 {{'_readonly' attribute only applies to global variables and parameters}}
  return a[n];
}

// The attributes are not valid on class members.
struct C {
  // expected-error@+1 {{'_readwrite' attribute only applies to global variables and parameters}}
  int _readwrite m;

  // expected-error@+2 {{'_writeonly' attribute only applies to global variables and parameters}}
  // expected-error@+1 {{field 'f' declared as a function}}
  void _writeonly f(int);
};

// The attributes are not valid on local variables.
void f9() {
  int _readonly ro;
  int _readwrite rw;
  int _writeonly wo;
  // expected-error@-3 {{'_readonly' attribute only applies to global variables and parameters}}
  // expected-error@-3 {{'_readwrite' attribute only applies to global variables and parameters}}
  // expected-error@-3 {{'_writeonly' attribute only applies to global variables and parameters}}
}
