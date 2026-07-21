//===- memory.cpp - Kitsune runtime memory allocation/free support --------===//
//
// Copyright (c) 2023, Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2023. Los Alamos National Security, LLC. This software was
//  produced under U.S. Government contract DE-AC52-06NA25396 for Los
//  Alamos National Laboratory (LANL), which is operated by Los Alamos
//  National Security, LLC for the U.S. Department of Energy. The
//  U.S. Government has rights to use, reproduce, and distribute this
//  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
//  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
//  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
//  derivative works, such modified software should be clearly marked,
//  so as not to confuse it with the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
//  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#include "common/logging.h"
#include "kitrt.h"
#include "memory_map.h"

#include <cstdlib>
#include <cstring>
#include <string>

template <typename T> static constexpr const char *getTypeName();
template <> constexpr const char *getTypeName<bool>() { return "bool"; }
template <> constexpr const char *getTypeName<int8_t>() { return "int8_t"; }
template <> constexpr const char *getTypeName<int16_t>() { return "int16_t"; }
template <> constexpr const char *getTypeName<int32_t>() { return "int32_t"; }
template <> constexpr const char *getTypeName<int64_t>() { return "int64_t"; }
template <> constexpr const char *getTypeName<float>() { return "float"; }
template <> constexpr const char *getTypeName<double>() { return "double"; }

template <typename T> static constexpr const char *getTypeFmt();
template <> constexpr const char *getTypeFmt<bool>() { return "%d"; }
template <> constexpr const char *getTypeFmt<int8_t>() { return "%d"; }
template <> constexpr const char *getTypeFmt<int16_t>() { return "%d"; }
template <> constexpr const char *getTypeFmt<int32_t>() { return "%d"; }
template <> constexpr const char *getTypeFmt<int64_t>() { return "%ld"; }
template <> constexpr const char *getTypeFmt<float>() { return "%f"; }
template <> constexpr const char *getTypeFmt<double>() { return "%g"; }

extern "C" [[gnu::malloc]] void *__kitrt_default_mem_alloc(uint64_t bytes) {
  void *ptr = malloc(bytes);
  __kitrt_register_mem_alloc(ptr, bytes);
  return ptr;
}

extern "C" void __kitrt_default_mem_free(void *ptr) {
  bool ro, wo;
  if (__kitrt_get_mem_alloc_size(ptr, &ro, &wo) > 0)
    __kitrt_unregister_mem_alloc(ptr);
  free(ptr);
}

template <typename T,
          std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>,
                           int> = 0>
static void mobileInitScalar(T *buf, size_t n, T v) {
  LOG("Setting %ld elements of mobile buffer with type '%s' to %s", n,
      getTypeName<T>(), std::to_string(v).c_str());
  for (size_t i = 0; i < n; ++i)
    buf[i] = v;
}

/**
 * Initialize a mobile buffer from a pointer to a value. This is most useful
 * when the buffer \p buf is a contiguous array of \p n \p size-byte objects.
 * This function will copy the object pointed to by \p v into each element of
 * \p buf.
 *
 * WARNING: This is a _shallow_ copy. For non-scalar types, this should only be
 * used with POD (Plain Old Data) types.
 *
 * NOTE: If the value to be copied is an integral or floating point type, it
 * may be more efficient to use one of the __kitrt_mobile_init_* functions
 * instead of this.
 */
extern "C" void __kitrt_mobile_init_from(void *buf, size_t n, void *v,
                                         unsigned size) {
  for (size_t i = 0; i < n; ++i) {
    memcpy(&((char *)buf)[i * size], v, size);
  }
}

/**
 * Initialize a mobile buffer \p buf. This is most useful when \p buf is a
 * contiguous array of \p n boolean values. This will initialize each element of
 * \p buf with \p v.
 */
extern "C" void __kitrt_mobile_init_bool(bool *buf, size_t n, bool v) {
  return mobileInitScalar<bool>(buf, n, v);
}

/**
 * Initialize a mobile buffer \p buf. This is most useful when \p buf is a
 * contiguous array of \p n 1-byte values. This will initialize each element of
 * \p buf with \p v.
 */
extern "C" void __kitrt_mobile_init_i8(int8_t *buf, size_t n, int8_t v) {
  return mobileInitScalar<int8_t>(buf, n, v);
}

/**
 * Initialize a mobile buffer \p buf. This is most useful when \p buf is a
 * contiguous array of \p n 2-byte values. This will initialize each element of
 * \p buf with \p v.
 */
extern "C" void __kitrt_mobile_init_i16(int16_t *buf, size_t n, int16_t v) {
  return mobileInitScalar<int16_t>(buf, n, v);
}

/**
 * Initialize a mobile buffer \p buf. This is most useful when \p buf is a
 * contiguous array of \p n 4-byte values. This will initialize each element of
 * \p buf with \p v.
 */
extern "C" void __kitrt_mobile_init_i32(int32_t *buf, size_t n, int32_t v) {
  return mobileInitScalar<int32_t>(buf, n, v);
}

/**
 * Initialize a mobile buffer \p buf. This is most useful when \p buf is a
 * contiguous array of \p n 8-byte values. This will initialize each element of
 * \p buf with \p v.
 */
extern "C" void __kitrt_mobile_init_i64(int64_t *buf, size_t n, int64_t v) {
  return mobileInitScalar<int64_t>(buf, n, v);
}

/**
 * Initialize a mobile buffer \p buf. This is most useful when \p buf is a
 * contiguous array of \p n 4-byte floats. This will initialize each element of
 * \p buf with \p v.
 */
extern "C" void __kitrt_mobile_init_float(float *buf, size_t n, float v) {
  return mobileInitScalar<float>(buf, n, v);
}

/**
 * Initialize a mobile buffer \p buf. This is most useful when \p buf is a
 * contiguous array of \p n 8-byte doubles. This will initialize each element of
 * \p buf with \p v.
 */
extern "C" void __kitrt_mobile_init_double(double *buf, size_t n, double v) {
  return mobileInitScalar<double>(buf, n, v);
}
