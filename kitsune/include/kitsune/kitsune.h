
/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */
#ifndef __KITSUNE_KITSUNE_H__
#define __KITSUNE_KITSUNE_H__

#include <stddef.h>
#include <stdint.h>

#if defined(spawn)
// FIXME KITSUNE: Should this be an error instead of a warning?
#warning encountered multiple definitions of spawn!
#else
#define spawn _kitsune_spawn
#endif

#if defined(sync)
// FIXME KITSUNE: Should this be an error instead of a warning?
#warning encountered multiple definitions of sync!
#else
#define sync _kitsune_sync
#endif

#if defined(forall)
// FIXME KITSUNE: Should this be an error instead of a warning?
#warning encountered multiple definitions of forall!
#else
#define forall _kitsune_forall
#endif

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif // __cplusplus

/// Allocate n bytes in a mobile buffer. This is unlikely to ever have a
/// definition. Instead, the compiler will replace calls to this with a call to
/// a suitable kitrt allocation function.
/// @param n The number of bytes to allocate.
/// @return A mobile pointer to the allocated buffer.
EXTERN_C void *__attribute__((malloc)) __kitrt_mobile_alloc(size_t n);

/// Deallocate a mobile buffer that was previously allocated. This is unlikely
/// to ever have a definition. Instead, the compiler will replace calls to this
/// with a call to a suitable kitrt deallocation function.
/// @param ptr The pointer to the buffer to be deallocated
EXTERN_C void __kitrt_mobile_free(void *ptr);

/// Allocate a mobile buffer with the given size in bytes.
EXTERN_C inline void *__attribute__((malloc))
kitsune_mobile_alloc(size_t bytes) {
  return __kitrt_mobile_alloc(bytes);
}

/// Deallocate (free) a mobile buffer that was previously allocated with
/// kitsune_mobile_alloc
EXTERN_C inline void kitsune_mobile_free(void *ptr) {
  return __kitrt_mobile_free(ptr);
}

#ifdef __cplusplus

namespace kitsune {
// TODO: This should not return a plain T*, but a __mobile__ T*.
/// Allocate a mobile buffer with n elements of the given type.
template <typename T>
inline T *__attribute__((malloc)) mobile_alloc(size_t n = 1) {
  return (T *)__kitrt_mobile_alloc(sizeof(T) * n);
}

// TODO: This should not return a plain T*, but a __mobile__ T*.
/// Deallocate a mobile buffer.
template <typename T> inline void mobile_free(T *ptr) {
  __kitrt_mobile_free(ptr);
}

/// A mobile pointer.
///
/// Mobile pointers are simply pointers to buffers whose contents may be moved
/// between devices. The data could be copied explicitly to and from host
/// memory and device (usually GPU) memory, or it could be done automatically,
/// for instance by allocating the data using UVM.
///
/// This is simply a wrapper around a raw pointer and does not enforce
/// additional semantics like an std::unique_ptr or an std::shared_ptr might.
/// This is intentional because this type is intended to allow for a somewhat
/// less invasive port from vanilla C++ to something that Kitsune can exploit.
/// However, this cannot be used interchangeably with raw pointers, or any
/// of the C++ smart pointers. The intention is to require the programmer to
/// explicitly annotate buffers that are mobile. In order to maintain
/// semantic similarity with raw pointers, the destructor will not free the
/// allocated pointer (if any). That must be done explicitly, otherwise, as
/// with regular pointers, memory may be leaked.
///
/// If the mobile type attribute is eventually added, users should prefer to
/// use that, but there is no guarantee that that attribute will be portable.
/// This, on the other hand, is guaranteed to be portable across C++
/// compilers.
///
template <typename T> class mobile_ptr {
public:
  using element_type = T;
  using pointer = element_type *;
  using reference_type = element_type &;

public:
  mobile_ptr() = default;
  mobile_ptr(mobile_ptr &) = default;
  mobile_ptr(mobile_ptr &&) = default;

  mobile_ptr &operator=(const mobile_ptr &o) {
    this->ptr = o.ptr;
    return *this;
  }

  /// Allocate a mobile_ptr buffer large enough to hold n elements of the
  /// contained type.
  mobile_ptr(size_t n) { alloc(n); }

  // TODO: This should not return a plain T*, but a __mobile__ T*.
  /// Get a pointer to the raw data.
  inline T *get() noexcept { return ptr; }

  // TODO: This should not return a plain T*, but a __mobile__ T*.
  /// Get a pointer to the raw data.
  inline const T *get() const noexcept { return ptr; }

  /// Allocate space for n elements. If an explicit number of elements is not
  /// provided, space for a single element will be allocated.
  void alloc(size_t n = 1) {
    if (ptr)
      this->free();
    ptr = mobile_alloc<T>(n);
  }

  /// Free the allocated mobile_ptr buffer. If the buffer has not already been
  /// allocated, this will have no effect. In either case, the contained
  /// pointer will be set to nullptr.
  void free() {
    mobile_free(ptr);
    ptr = nullptr;
  }

  /// Check whether the contained pointer has been allocated.
  inline operator bool() const noexcept { return ptr; }

  /// Dereference the contained raw pointer. The raw pointer must have been
  /// allocated.
  inline element_type &operator*() const { return *this->get(); }

  /// Access the object pointed to by the contained raw pointer. The pointer
  /// must have been allocated.
  inline pointer operator->() const { return get(); }

  /// Access the i'th element in the allocated array. This is only intended
  /// to be used when an actual array is allocated.
  inline T &operator[](size_t i) { return ptr[i]; }

  /// Access the i'th element in the allocated array. This is only intended
  /// to be used when an actual array is allocated.
  inline const T &operator[](size_t i) const { return ptr[i]; }

private:
  // TODO: This should not be a plain T*, but a __mobile__ T*.
  /// The raw pointer.
  pointer ptr = nullptr;
};
} // namespace kitsune

#endif // ! __cplusplus

#undef EXTERN_C

#endif // __KITSUNE_KITSUNE_H__
