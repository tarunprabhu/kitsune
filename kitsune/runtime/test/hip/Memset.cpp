// REQUIRES: amd-gpu
//
// Check that Kitsune's memset wrappers work as expected.
//
// This test does not produce any output, but will return an error code on
// failure.
//
// RUN: %exe

#include "common/unreachable.h"
#include "hip/kithip.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstdlib>

template <typename T> static T min() { return std::numeric_limits<T>::min(); }

template <typename T> static T max() { return std::numeric_limits<T>::max(); }

template <typename T> static unsigned test(unsigned n, T init) {
  T *buf = (T *)__kithip_malloc(n * sizeof(T));
  if constexpr (std::is_same_v<T, bool>)
    __kithip_memset_bool(buf, n, init);
  else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
    __kithip_memset_i8(buf, n, init);
  else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>)
    __kithip_memset_i16(buf, n, init);
  else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>)
    __kithip_memset_i32(buf, n, init);
  else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
    __kithip_memset_i64(buf, n, init);
  else if constexpr (std::is_same_v<T, float>)
    __kithip_memset_float(buf, n, init);
  else if constexpr (std::is_same_v<T, double>)
    __kithip_memset_double(buf, n, init);
  else
    UNREACHABLE("Unsupported type");

  T *res = new T[n];
  __kithip_memcpy_dtoh(res, buf, n * sizeof(T));

  unsigned errs = 0;
  for (unsigned i = 0; i < n; ++i)
    if (res[i] != init)
      errs += 1;

  delete[] res;
  __kithip_free(buf);

  return errs;
}

int main(int argc, char *argv[]) {
  unsigned n = 8191;
  if (argc > 1)
    n = atol(argv[1]);

  long errs = 0;
  errs += test<bool>(n, true);
  errs += test<int8_t>(n, min<int8_t>() + 1);
  errs += test<uint8_t>(n, max<uint8_t>() - 1);
  errs += test<int16_t>(n, min<int16_t>() + 1);
  errs += test<uint16_t>(n, max<uint16_t>() - 1);
  errs += test<int32_t>(n, min<int32_t>() + 1);
  errs += test<uint32_t>(n, max<uint32_t>() - 1);
  errs += test<int64_t>(n, min<int64_t>() + 1);
  errs += test<uint64_t>(n, max<uint64_t>() - 1);
  errs += test<float>(n, 3.14);
  errs += test<double>(n, 2.71828);

  // TODO: Add check for __kithip_memset_from.

  // Don't return the number of errors because, depending on the system, only
  // the last 7 bits may be examined for success/failure.
  return errs != 0;
}
