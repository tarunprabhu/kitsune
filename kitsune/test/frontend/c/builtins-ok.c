// Check that the correct use of Kitsune's builtins does not produce any
// diagnostics.
//
// RUN: %kitcc -std=c23 -Xclang -verify -fsyntax-only %s
//
// expected-no-diagnostics

#include <kitsune.h>

void *__attribute__((kitsune_mobile)) allocate(unsigned long n) {
  return kitsune_mobile_alloc(n);
}

void deallocate(void *[[kitsune::mobile]] ptr) {
  kitsune_mobile_free(ptr);
}

void *[[kitsune::mobile]] cast_unsafe(void *ptr) {
  return __kitsune_mobile_cast_unsafe(ptr);
}

void reduce_band() {
  int8_t i8;
  int16_t i16;
  int32_t i32;
  int64_t i64;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  __kitsune_reduce(&i8, KIT_BAND, i8);
  __kitsune_reduce(&u8, KIT_BAND, u8);
  __kitsune_reduce(&i16, KIT_BAND, i16);
  __kitsune_reduce(&u16, KIT_BAND, u16);
  __kitsune_reduce(&i32, KIT_BAND, i32);
  __kitsune_reduce(&u32, KIT_BAND, u32);
  __kitsune_reduce(&i64, KIT_BAND, i64);
  __kitsune_reduce(&u64, KIT_BAND, u64);
}

void reduce_bor() {
  int8_t i8;
  int16_t i16;
  int32_t i32;
  int64_t i64;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  __kitsune_reduce(&i8, KIT_BOR, i8);
  __kitsune_reduce(&u8, KIT_BOR, u8);
  __kitsune_reduce(&i16, KIT_BOR, i16);
  __kitsune_reduce(&u16, KIT_BOR, u16);
  __kitsune_reduce(&i32, KIT_BOR, i32);
  __kitsune_reduce(&u32, KIT_BOR, u32);
  __kitsune_reduce(&i64, KIT_BOR, i64);
  __kitsune_reduce(&u64, KIT_BOR, u64);
}

void reduce_bxor() {
  int8_t i8;
  int16_t i16;
  int32_t i32;
  int64_t i64;
  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;

  __kitsune_reduce(&i8, KIT_BXOR, i8);
  __kitsune_reduce(&u8, KIT_BXOR, u8);
  __kitsune_reduce(&i16, KIT_BXOR, i16);
  __kitsune_reduce(&u16, KIT_BXOR, u16);
  __kitsune_reduce(&i32, KIT_BXOR, i32);
  __kitsune_reduce(&u32, KIT_BXOR, u32);
  __kitsune_reduce(&i64, KIT_BXOR, i64);
  __kitsune_reduce(&u64, KIT_BXOR, u64);
}

void reduce_land() {
  bool b;
  __kitsune_reduce(&b, KIT_LAND, b);
}

void reduce_lor() {
  bool b;
  __kitsune_reduce(&b, KIT_LOR, b);
}

void reduce_lxor() {
  bool b;
  __kitsune_reduce(&b, KIT_LXOR, b);
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

void reduce_prod() {
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

  __kitsune_reduce(&i8, KIT_PROD, i8);
  __kitsune_reduce(&u8, KIT_PROD, u8);
  __kitsune_reduce(&i16, KIT_PROD, i16);
  __kitsune_reduce(&u16, KIT_PROD, u16);
  __kitsune_reduce(&i32, KIT_PROD, i32);
  __kitsune_reduce(&u32, KIT_PROD, u32);
  __kitsune_reduce(&i64, KIT_PROD, i64);
  __kitsune_reduce(&u64, KIT_PROD, u64);
  __kitsune_reduce(&f32, KIT_PROD, f32);
  __kitsune_reduce(&f64, KIT_PROD, f64);
}

void reduce_sum() {
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

  __kitsune_reduce(&i8, KIT_SUM, i8);
  __kitsune_reduce(&u8, KIT_SUM, u8);
  __kitsune_reduce(&i16, KIT_SUM, i16);
  __kitsune_reduce(&u16, KIT_SUM, u16);
  __kitsune_reduce(&i32, KIT_SUM, i32);
  __kitsune_reduce(&u32, KIT_SUM, u32);
  __kitsune_reduce(&i64, KIT_SUM, i64);
  __kitsune_reduce(&u64, KIT_SUM, u64);
  __kitsune_reduce(&f32, KIT_SUM, f32);
  __kitsune_reduce(&f64, KIT_SUM, f64);
}
