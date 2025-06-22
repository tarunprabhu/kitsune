// RUN: %kitxx -Xclang -verify -fsyntax-only -ftapir=serial %s
// RUN: %kitcc -x c -std=c23 -Xclang -verify -fsyntax-only -ftapir=serial %s
// expected-no-diagnostics

void f1(int *[[kitsune::mobile]] ptr) {}
void f2(void *__attribute__((kitsune_mobile)) ptr) {}
