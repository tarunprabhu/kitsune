// RUN: %kitcc --tapir=nolo -Xclang -verify -fsyntax-only %s

// Attributes are not valid on functions.
// expected-error-re@+1 {{'_readonly' attribute only applies to {{.+}}}}
int _readonly f5(int* a, int n);

// expected-error-re@+1 {{'_readwrite' attribute only applies to {{.+}}}}
int _readwrite f6(int* a, int n);

// expected-error-re@+1 {{'_writeonly' attribute only applies to {{.+}}}}
int _writeonly f7(int* a, int n);

// expected-error-re@+1 {{'_readonly' attribute only applies to {{.+}}}}
int _readonly f8(int* a, int n) {
  return a[n];
}

// The attributes are not valid on class members.
struct C {
  // expected-error-re@+1 {{'_readwrite' attribute only applies to {{.+}}}}
  int _readwrite m;

  // expected-error-re@+2 {{'_writeonly' attribute only applies to {{.+}}}}
  // expected-error@+1 {{field 'f' declared as a function}}
  void _writeonly f(int);
};

// The attributes are not valid on local variables.
void f9() {
  // expected-error-re@+1 {{'_readonly' attribute only applies to {{.+}}}}
  int _readonly ro;

  // expected-error-re@+1 {{'_readwrite' attribute only applies to {{.+}}}}
  int _readwrite rw;

  // expected-error-re@+1 {{'_writeonly' attribute only applies to {{.+}}}}
  int _writeonly wo;
}
