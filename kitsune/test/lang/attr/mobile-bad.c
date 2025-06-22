// RUN: %kitcc -std=c23 -Xclang -verify -ftapir=serial -fsyntax-only %s

#include <stdlib.h>

typedef struct {
  int n;
} S1;

int [[kitsune::mobile]] i;
// expected-error@-1 {{'kitsune::mobile' can only be used on a pointer type}}

S1 [[kitsune::mobile]] s1;
// expected-error@-1 {{'kitsune::mobile' can only be used on a pointer type}}

double [[kitsune::mobile]] dbl[3];
// expected-error@-1 {{'kitsune::mobile' can only be used on a pointer type}}

void (*fptr)(int) [[kitsune::mobile]];
// expected-error@-1 {{'kitsune::mobile' can only be used on a pointer type}}

int fn(int) [[kitsune::mobile]];
// expected-error@-1 {{'kitsune::mobile' can only be used on a pointer type}}

int f1(int* ptr) {
  return *ptr;
}

int f2(int* [[kitsune::mobile]] ptr) {
  // expected-error-re@+1 {{passing {{.+}} to parameter {{.+}} discards qualifiers}}
  return f1(ptr);
  // expected-note@-7 {{passing argument to parameter}}
}

int* f3(int* [[kitsune::mobile]] ptr) {
  // expected-error-re@+1 {{returning {{.+}} from a function {{.+}} discards qualifiers}}
  return ptr;
}

void f4(int* [[kitsune::mobile]] arg) {
  // expected-error-re@+1 {{initializing {{.+}} with an expression {{.+}} discards qualifiers}}
  int* local = arg;
}

void f5() {
  int* [[kitsune::mobile]] ptr = NULL;
  int* local = NULL;

  // expected-error-re@+1 {{assigning to {{.+}} discards qualifiers}}
  local = ptr;
}

int* g6 = NULL;
void f6(int* [[kitsune::mobile]] ptr) {
  // expected-error-re@+1 {{assigning to {{.+}} discards qualifiers}}
  g6 = ptr;
}

float* f7(float* [[kitsune::mobile]] ptr, int i) {
  // expected-error-re@+1 {{returning {{.+}} from a function {{.+}} discards qualifiers}}
  return &ptr[i];
}

void f8(float* [[kitsune::mobile]] ptr, int i) {
  // expected-error-re@+1 {{initializing {{.+}} with an expression {{.+}} discards qualifiers}}
  float* local = &ptr[i];
}
