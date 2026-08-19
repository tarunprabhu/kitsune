// RUN: %kitxx -Xclang -verify -fsyntax-only -ferror-limit=0 %s

#include <kitsune.h>

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

void reduce_num_args(int a, int n) {
  // expected-error@+1 {{incorrect number of arguments}}
  __kitsune_reduce(&a, KIT_ADD);

  // expected-error@+1 {{incorrect number of arguments}}
  __kitsune_reduce(&a, KIT_ADD, n, nullptr);
}

void reduce_non_literal_op(int a, int n, unsigned op) {
  // expected-error@+1 {{reduction operator must be an integer literal}}
  __kitsune_reduce(&a, op, n);

  // expected-error@+1 {{cannot initialize a parameter of type}}
  __kitsune_reduce(&a, "+=", n);
}

void reduce_unknown_op(int a, int n) {
  // expected-error@+1 {{reduction operator must be an integer literal}}
  __kitsune_reduce(&a, -1, n);

  // expected-error@+1 {{unknown reduction operator}}
  __kitsune_reduce(&a, 0xFFFFFFFF, n);

  // expected-error@+1 {{unknown reduction operator}}
  __kitsune_reduce(&a, 15, n);
}

void reduce_non_scalar() {
  struct Complex { float r; float i; } c;
  enum Response { RESP_YES, RESP_NO, RESP_MAYBE } e;
  int *ptr;

  // expected-error-re@+1 {{value {{.+}} must have builtin scalar type}}
  __kitsune_reduce(&e, KIT_ADD, e);

  // expected-error-re@+1 {{value {{.+}} must have builtin scalar type}}
  __kitsune_reduce(&ptr, KIT_MIN, ptr);

  // expected-error-re@+1 {{value {{.+}} must have builtin scalar type}}
  __kitsune_reduce(&c, KIT_MUL, c);
}

void reduce_type_mismatch(void *ptr) {
  int8_t i8;
  uint8_t u8;
  int32_t i32;
  uint32_t u32;
  float f32;
  int32_t *pi32;

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(ptr, KIT_ADD, i8);

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(&i8, KIT_MUL, u8);

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(&u8, KIT_MIN, i8);

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(&i32, KIT_AND, u32);

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(&u32, KIT_OR, i32);

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(&i32, KIT_MAX, i8);

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(&f32, KIT_XOR, i32);

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(&pi32, KIT_ADD, i32);

  // Force casting is allowed, obviously at the user's own peril.
  __kitsune_reduce((uint32_t*)&i32, KIT_AND, u32);
  __kitsune_reduce((int32_t*)&i8, KIT_AND, i32);
  __kitsune_reduce(&i8, KIT_MUL, (int8_t)f32);
}

void reduce_incompatible_and() {
  double d;
  float f;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'float'}}
  __kitsune_reduce(&f, KIT_AND, f);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'double'}}
  __kitsune_reduce(&d, KIT_AND, d);
}

void reduce_incompatible_or() {
  double d;
  float f;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'float'}}
  __kitsune_reduce(&f, KIT_OR, f);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'double'}}
  __kitsune_reduce(&d, KIT_OR, d);
}

void reduce_incompatible_xor() {
  double d;
  float f;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'float'}}
  __kitsune_reduce(&f, KIT_XOR, f);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'double'}}
  __kitsune_reduce(&d, KIT_XOR, d);
}

void reduce_incompatible_max() {
  bool b;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'bool'}}
  __kitsune_reduce(&b, KIT_MAX, b);
}

void reduce_incompatible_maximum() {
  bool b;
  int i;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'bool'}}
  __kitsune_reduce(&b, KIT_MAXIMUM, b);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'int'}}
  __kitsune_reduce(&i, KIT_MAXIMUM, i);
}

void reduce_incompatible_maximum_num() {
  bool b;
  int i;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'bool'}}
  __kitsune_reduce(&b, KIT_MAXIMUM_NUM, b);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'int'}}
  __kitsune_reduce(&i, KIT_MAXIMUM_NUM, i);
}

void reduce_incompatible_min() {
  bool b;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'bool'}}
  __kitsune_reduce(&b, KIT_MIN, b);
}

void reduce_incompatible_minimum() {
  bool b;
  int i;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'bool'}}
  __kitsune_reduce(&b, KIT_MINIMUM, b);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'int'}}
  __kitsune_reduce(&i, KIT_MINIMUM, i);
}

void reduce_incompatible_minimum_num() {
  bool b;
  int i;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'bool'}}
  __kitsune_reduce(&b, KIT_MINIMUM_NUM, b);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'int'}}
  __kitsune_reduce(&i, KIT_MINIMUM_NUM, i);
}

void reduce_incompatible_mul() {
  bool b;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'bool'}}
  __kitsune_reduce(&b, KIT_MUL, b);
}

void reduce_incompatible_add() {
  bool b;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'bool'}}
  __kitsune_reduce(&b, KIT_ADD, b);
}

void reduce_unsupported() {
  int64_t i64;

  // expected-error-re@+1 {{{{.+}} operator '{{.+}}' is not yet supported}}
  __kitsune_reduce(&i64, KIT_CUSTOM, i64);
}
