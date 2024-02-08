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
  __kitsune_reduce(&a, KIT_SUM);

  // expected-error@+1 {{incorrect number of arguments}}
  __kitsune_reduce(&a, KIT_SUM, n, nullptr);
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
  __kitsune_reduce(&a, 13, n);
}

void reduce_non_scalar() {
  struct Complex { float r; float i; } c;
  enum Response { RESP_YES, RESP_NO, RESP_MAYBE } e;
  int *ptr;

  // expected-error-re@+1 {{value {{.+}} must have builtin scalar type}}
  __kitsune_reduce(&e, KIT_SUM, e);

  // expected-error-re@+1 {{value {{.+}} must have builtin scalar type}}
  __kitsune_reduce(&ptr, KIT_MIN, ptr);

  // expected-error-re@+1 {{value {{.+}} must have builtin scalar type}}
  __kitsune_reduce(&c, KIT_PROD, c);
}

void reduce_type_mismatch(void *ptr) {
  int8_t i8;
  uint8_t u8;
  int32_t i32;
  uint32_t u32;
  float f32;
  int32_t *pi32;

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(ptr, KIT_SUM, i8);

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(&i8, KIT_PROD, u8);

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(&u8, KIT_MIN, i8);

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(&i32, KIT_BAND, u32);

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(&u32, KIT_BOR, i32);

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(&i32, KIT_MAX, i8);

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(&f32, KIT_BXOR, i32);

  // expected-error-re@+1 {{{{.+}} destination must be pointer to {{.+}}}}
  __kitsune_reduce(&pi32, KIT_SUM, i32);

  // Force casting is allowed, obviously at the user's own peril.
  __kitsune_reduce((uint32_t*)&i32, KIT_BAND, u32);
  __kitsune_reduce((int32_t*)&i8, KIT_BAND, i32);
  __kitsune_reduce(&i8, KIT_PROD, (int8_t)f32);
}

void reduce_incompatible_band() {
  double d;
  float f;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'float'}}
  __kitsune_reduce(&f, KIT_BAND, f);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'double'}}
  __kitsune_reduce(&d, KIT_BAND, d);
}

void reduce_incompatible_bor() {
  double d;
  float f;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'float'}}
  __kitsune_reduce(&f, KIT_LOR, f);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'double'}}
  __kitsune_reduce(&d, KIT_BOR, d);
}

void reduce_incompatible_bxor() {
  double d;
  float f;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'float'}}
  __kitsune_reduce(&f, KIT_BXOR, f);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'double'}}
  __kitsune_reduce(&d, KIT_BXOR, d);
}

void reduce_incompatible_land() {
  double d;
  float f;
  long l;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'long'}}
  __kitsune_reduce(&l, KIT_LAND, l);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'float'}}
  __kitsune_reduce(&f, KIT_LAND, f);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'double'}}
  __kitsune_reduce(&d, KIT_LAND, d);
}

void reduce_incompatible_lor() {
  double d;
  float f;
  long l;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'long'}}
  __kitsune_reduce(&l, KIT_LOR, l);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'float'}}
  __kitsune_reduce(&f, KIT_LOR, f);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'double'}}
  __kitsune_reduce(&d, KIT_LOR, d);
}

void reduce_incompatible_lxor() {
  double d;
  float f;
  long l;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'long'}}
  __kitsune_reduce(&l, KIT_LXOR, l);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'float'}}
  __kitsune_reduce(&f, KIT_LXOR, f);

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'double'}}
  __kitsune_reduce(&d, KIT_LXOR, d);
}

void reduce_incompatible_max() {
  bool b;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'bool'}}
  __kitsune_reduce(&b, KIT_MAX, b);
}

void reduce_incompatible_min() {
  bool b;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'bool'}}
  __kitsune_reduce(&b, KIT_MIN, b);
}

void reduce_incompatible_prod() {
  bool b;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'bool'}}
  __kitsune_reduce(&b, KIT_PROD, b);
}

void reduce_incompatible_sum() {
  bool b;

  // expected-error-re@+1 {{{{.+}} operator {{.+}} not valid for {{.+}} 'bool'}}
  __kitsune_reduce(&b, KIT_SUM, b);
}

void reduce_unsupported() {
  int64_t i64;

  // expected-error-re@+1 {{{{.+}} operator '{{.+}}' is not yet supported}}
  __kitsune_reduce(&i64, KIT_MAXLOC, i64);

  // expected-error-re@+1 {{{{.+}} operator '{{.+}}' is not yet supported}}
  __kitsune_reduce(&i64, KIT_MINLOC, i64);

  // expected-error-re@+1 {{{{.+}} operator '{{.+}}' is not yet supported}}
  __kitsune_reduce(&i64, KIT_CUSTOM, i64);
}
