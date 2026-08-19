// Check that the correct use of Kitsune's builtins does not produce any
// diagnostics.
//
// RUN: %kitxx -Xclang -verify -fsyntax-only %s
//
// expected-no-diagnostics

#include <kitsune.h>

void *__attribute__((kitsune_mobile)) allocate(unsigned long n) {
  return kitsune_mobile_alloc(n);
}

void deallocate(void *[[kitsune::mobile]] ptr) { kitsune_mobile_free(ptr); }

void *[[kitsune::mobile]] cast_unsafe(void *ptr) {
  return __kitsune_mobile_cast_unsafe(ptr);
}

void reduce_and() {
  int8_t i8;
  int16_t i16;
  int32_t i32;
  int64_t i64;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  __kitsune_reduce(&i8, KIT_AND, i8);
  __kitsune_reduce(&u8, KIT_AND, u8);
  __kitsune_reduce(&i16, KIT_AND, i16);
  __kitsune_reduce(&u16, KIT_AND, u16);
  __kitsune_reduce(&i32, KIT_AND, i32);
  __kitsune_reduce(&u32, KIT_AND, u32);
  __kitsune_reduce(&i64, KIT_AND, i64);
  __kitsune_reduce(&u64, KIT_AND, u64);
}

void reduce_or() {
  int8_t i8;
  int16_t i16;
  int32_t i32;
  int64_t i64;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  __kitsune_reduce(&i8, KIT_OR, i8);
  __kitsune_reduce(&u8, KIT_OR, u8);
  __kitsune_reduce(&i16, KIT_OR, i16);
  __kitsune_reduce(&u16, KIT_OR, u16);
  __kitsune_reduce(&i32, KIT_OR, i32);
  __kitsune_reduce(&u32, KIT_OR, u32);
  __kitsune_reduce(&i64, KIT_OR, i64);
  __kitsune_reduce(&u64, KIT_OR, u64);
}

void reduce_xor() {
  int8_t i8;
  int16_t i16;
  int32_t i32;
  int64_t i64;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  __kitsune_reduce(&i8, KIT_XOR, i8);
  __kitsune_reduce(&u8, KIT_XOR, u8);
  __kitsune_reduce(&i16, KIT_XOR, i16);
  __kitsune_reduce(&u16, KIT_XOR, u16);
  __kitsune_reduce(&i32, KIT_XOR, i32);
  __kitsune_reduce(&u32, KIT_XOR, u32);
  __kitsune_reduce(&i64, KIT_XOR, i64);
  __kitsune_reduce(&u64, KIT_XOR, u64);
}

void reduce_max() {
  int8_t i8;
  int16_t i16;
  int32_t i32;
  int64_t i64;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;
  float f32;
  double f64;

  __kitsune_reduce(&i8, KIT_MAX, i8);
  __kitsune_reduce(&u8, KIT_MAX, u8);
  __kitsune_reduce(&i16, KIT_MAX, i16);
  __kitsune_reduce(&u16, KIT_MAX, u16);
  __kitsune_reduce(&i32, KIT_MAX, i32);
  __kitsune_reduce(&u32, KIT_MAX, u32);
  __kitsune_reduce(&i64, KIT_MAX, i64);
  __kitsune_reduce(&u64, KIT_MAX, u64);
  __kitsune_reduce(&f32, KIT_MAX, f32);
  __kitsune_reduce(&f64, KIT_MAX, f64);
}

void reduce_maximum() {
  float f32;
  double f64;

  __kitsune_reduce(&f32, KIT_MAXIMUM, f32);
  __kitsune_reduce(&f64, KIT_MAXIMUM, f64);
}

void reduce_maximum_num() {
  float f32;
  double f64;

  __kitsune_reduce(&f32, KIT_MAXIMUM_NUM, f32);
  __kitsune_reduce(&f64, KIT_MAXIMUM_NUM, f64);
}

void reduce_min() {
  int8_t i8;
  int16_t i16;
  int32_t i32;
  int64_t i64;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;
  float f32;
  double f64;

  __kitsune_reduce(&i8, KIT_MIN, i8);
  __kitsune_reduce(&u8, KIT_MIN, u8);
  __kitsune_reduce(&i16, KIT_MIN, i16);
  __kitsune_reduce(&u16, KIT_MIN, u16);
  __kitsune_reduce(&i32, KIT_MIN, i32);
  __kitsune_reduce(&u32, KIT_MIN, u32);
  __kitsune_reduce(&i64, KIT_MIN, i64);
  __kitsune_reduce(&u64, KIT_MIN, u64);
  __kitsune_reduce(&f32, KIT_MIN, f32);
  __kitsune_reduce(&f64, KIT_MIN, f64);
}

void reduce_minimum() {
  float f32;
  double f64;

  __kitsune_reduce(&f32, KIT_MINIMUM, f32);
  __kitsune_reduce(&f64, KIT_MINIMUM, f64);
}

void reduce_minimum_num() {
  float f32;
  double f64;

  __kitsune_reduce(&f32, KIT_MINIMUM_NUM, f32);
  __kitsune_reduce(&f64, KIT_MINIMUM_NUM, f64);
}

void reduce_mul() {
  int8_t i8;
  int16_t i16;
  int32_t i32;
  int64_t i64;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;
  float f32;
  double f64;

  __kitsune_reduce(&i8, KIT_MUL, i8);
  __kitsune_reduce(&u8, KIT_MUL, u8);
  __kitsune_reduce(&i16, KIT_MUL, i16);
  __kitsune_reduce(&u16, KIT_MUL, u16);
  __kitsune_reduce(&i32, KIT_MUL, i32);
  __kitsune_reduce(&u32, KIT_MUL, u32);
  __kitsune_reduce(&i64, KIT_MUL, i64);
  __kitsune_reduce(&u64, KIT_MUL, u64);
  __kitsune_reduce(&f32, KIT_MUL, f32);
  __kitsune_reduce(&f64, KIT_MUL, f64);
}

void reduce_add() {
  int8_t i8;
  int16_t i16;
  int32_t i32;
  int64_t i64;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;
  float f32;
  double f64;

  __kitsune_reduce(&i8, KIT_ADD, i8);
  __kitsune_reduce(&u8, KIT_ADD, u8);
  __kitsune_reduce(&i16, KIT_ADD, i16);
  __kitsune_reduce(&u16, KIT_ADD, u16);
  __kitsune_reduce(&i32, KIT_ADD, i32);
  __kitsune_reduce(&u32, KIT_ADD, u32);
  __kitsune_reduce(&i64, KIT_ADD, i64);
  __kitsune_reduce(&u64, KIT_ADD, u64);
  __kitsune_reduce(&f32, KIT_ADD, f32);
  __kitsune_reduce(&f64, KIT_ADD, f64);
}
