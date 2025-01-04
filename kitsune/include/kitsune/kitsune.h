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

// TODO: This should be replaced with #ifndef __compile_is_kitsune__ which is
// only defined when the code is compiled with Kitsune, but not otherwise. This
// is allow code containing kitsune builtins and library functions to be
// compiled with another compiler and maintain "reasonable" behavior.
// #ifndef __compiler_is_kitsune
#if 0

/// Allocate n bytes in a mobile buffer. In Kitsune, this is a builtin that is
/// replaced with a suitable memory allocation function depending on the tapir
/// target(s) used. This is here if the code is not compiled with Kitsune, and
/// simply calls the system's default memory allocator (malloc).
/// \param  n The number of bytes to allocate.
/// \return The pointer ointer to the allocated buffer.
EXTERN_C inline void *__attribute__((malloc))
kitsune_mobile_alloc(size_t bytes) {
  return malloc(bytes);
}

/// Deallocate (free) a mobile buffer that was previously allocated with
/// kitsune_mobile_alloc. In Kitsune, this is an intrinsic that is replaced with
/// a call to an appropriate runtime function. This is here if the code is not
/// compiled with Kitsune and simply calls the system's default deallocator.
EXTERN_C inline void kitsune_mobile_free(void *ptr) {
  return free(ptr);
}

#endif // __compiler_is_kitsune

#ifdef __cplusplus

namespace kitsune {

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
/// This, on the other hand, is guaranteed to be portable across C++ compilers.
///
template <typename T> class mobile_ptr {
public:
  using element_type = T;

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

  /// Get a pointer to the raw data.
  inline T *[[kitsune::mobile]] get() noexcept { return ptr; }

  // TODO: This should not return a plain T*, but a __mobile__ T*.
  /// Get a pointer to the raw data.
  inline const T *[[kitsune::mobile]] get() const noexcept { return ptr; }

  /// Allocate space for n elements. If an explicit number of elements is not
  /// provided, space for a single element will be allocated.
  void alloc(size_t n = 1) {
    if (ptr)
      this->free();
    ptr = (T* [[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(T));
  }

  /// Free the allocated mobile_ptr buffer. If the buffer has not already been
  /// allocated, this will have no effect. In either case, the contained
  /// pointer will be set to nullptr.
  void free() {
    kitsune_mobile_free(ptr);
    ptr = nullptr;
  }

  /// Check whether the contained pointer has been allocated.
  inline operator bool() const noexcept { return ptr; }

  /// Dereference the contained raw pointer. The raw pointer must have been
  /// allocated.
  inline element_type &operator*() const { return *this->get(); }

  /// Access the object pointed to by the contained raw pointer. The pointer
  /// must have been allocated.
  inline T *operator->() const { return get(); }

  /// Access the i'th element in the allocated array. This is only intended
  /// to be used when an actual array is allocated.
  inline T &operator[](size_t i) { return ptr[i]; }

  /// Access the i'th element in the allocated array. This is only intended
  /// to be used when an actual array is allocated.
  inline const T &operator[](size_t i) const { return ptr[i]; }

private:
  /// The raw pointer.
  T *[[kitsune::mobile]] ptr = nullptr;
};

} // namespace kitsune

#endif // ! __cplusplus

#undef EXTERN_C

#endif // __KITSUNE_KITSUNE_H__
