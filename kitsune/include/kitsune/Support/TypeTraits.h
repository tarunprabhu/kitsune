//===- TypeTraits.h - Type traits ------------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convenience type traits that are not part of the standard headers.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_TYPE_TRAITS_H
#define KITSUNE_SUPPORT_TYPE_TRAITS_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

// These are added to the std namespace for consistency with other standard
// traits.
namespace std {

namespace detail {

template <typename T, typename = void>
struct is_iterable_t : std::false_type {};

template <typename T>
struct is_iterable_t<T, std::void_t<decltype(std::declval<T>().begin()),
                                    decltype(std::declval<T>().end())>>
    : std::true_type {};

template <typename T, template <typename, unsigned, typename...> class C>
struct is_small_adt : std::false_type {};

template <typename E, unsigned N, typename... Extra,
          template <typename, unsigned, typename...> class C>
struct is_small_adt<C<E, N, Extra...>, C> : std::true_type {};

template <typename T, template <unsigned> class C>
struct is_small_string : std::false_type {};

template <unsigned N, template <unsigned> class C>
struct is_small_string<C<N>, C> : std::true_type {};

} // namespace detail

/// \addtogroup kitsune
/// @{

/// Return true if the type is iterable. This will be the case if the type has
/// begin() and end() members. This makes the type suitable to be used in a
/// range-based for loop.
template <typename T>
static constexpr bool is_iterable_v = detail::is_iterable_t<T>::value;

/// Return true if the type is a boolean. This will not return true if `T` is
/// bool&.
template <typename T>
static constexpr bool is_bool_v = std::is_same_v<std::remove_cv_t<T>, bool>;

/// Check if the type is an integer, but not a boolean. This will return true
/// for the following types and their aliases:
///
///     int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t
///
/// as well as the cv-qualified variants of these. This will not return true if
/// `T` is a reference type.
template <typename T>
static constexpr bool is_integer_v =
    std::is_integral_v<std::remove_cv_t<T>> && !std::is_bool_v<T>;

/// Check if the type is a 32-bit floating point type. This will return on the
/// following types and their aliases:
///
///     float
///
/// as well as the cv-qualified variants of these. This will not return true if
/// `T` is a reference type.
template <typename T>
static constexpr bool is_float_v = std::is_same_v<std::remove_cv_t<T>, float>;

/// Check if the type is a 64-bit floating point type. This will return on the
/// following types and their aliases:
///
///     double
///
/// as well as the cv-qualified variants of these. This will not return true if
/// `T` is a reference type.
template <typename T>
static constexpr bool is_double_v = std::is_same_v<std::remove_cv_t<T>, double>;

/// Check if the type is a long double type. This will return on the following
/// types and their aliases:
///
///     long double
///
/// as well as the cv-qualified variants of these. This will not return true if
/// `T` is a reference type.
template <typename T>
static constexpr bool is_long_double_v =
    std::is_same_v<std::remove_cv_t<T>, long double>;

/// Check if the type is an instance of LLVM's SmallString type.
template <typename C>
static constexpr bool is_small_string_v =
    detail::is_small_string<C, llvm::SmallString>::value;

/// Check if the type is an instance of LLVM's SmallSet type.
template <typename C>
static constexpr bool is_small_set_v =
    detail::is_small_adt<C, llvm::SmallSet>::value;

/// Check if the type is an instance of LLVM's SmallVector type.
template <typename C>
static constexpr bool is_small_vector_v =
    detail::is_small_adt<C, llvm::SmallVector>::value;

/// Check if the type is a string-like type. This will return true on the
/// following types and their aliases:
///
///     std::string, llvm::StringLiteral, llvm::StringRef, llvm::SmallString
///
/// as well as the cv-qualified variants of these. This will not return true if
/// `T` is a reference type.
template <typename T>
static constexpr bool is_string_like_v =
    std::is_same_v<std::remove_cv_t<T>, std::string> ||
    std::is_same_v<std::remove_cv_t<T>, llvm::StringLiteral> ||
    std::is_same_v<std::remove_cv_t<T>, llvm::StringRef> ||
    std::is_small_string_v<std::remove_cv_t<T>>;

/// Check if the type is "interoperable". In Kitsune, this refers to structs
/// that have a standard-layout, are trivially constructible, and (in C++
/// parlance) are aggregates. In practice, this means that they are contiguous
/// in memory, adhere to the C language layout rules, do not contain any
/// non-static private or protected members, and do not contain any
/// user-provided constructors (defaulted default constructors are permitted).
///
/// In Kitsune, this is used for types of objects that are created by the
/// compiler, but only consumed by the runtime. Essentially, these types are
/// serialized by the compiler and deserialized by the runtime.
///
/// All interoperable types *must* be structs. This implies that even if we
/// would like to pass a scalar type, it must be wrapped in a struct.
///
/// ---------------------------------- FIXME ----------------------------------
///
/// It is not clear if the constraints on accessibility of members is strictly
/// necessary. As long as the declaration of the type is shared between the
/// compiler and the runtime, serializing the an object in the compiler and
/// deserializing it in the runtime should not be a problem. Issues may arise if
/// the compiler and runtime were compiled with different compilers, thus
/// resulting in mismatched layouts. This behavior is not supported anyway. For
/// now, we only support a limited amount of interoperability, so the additional
/// constraints are not an issue, but they ought to be addressed at some point.
///
/// A major issue with this trait that it will claim that structs with pointer
/// types are interoperable. This is a problem because, for our use cases, such
/// types are most definitely *not* interoperable.
///
template <typename T>
static constexpr bool is_interop_v =
    std::is_class_v<T> && std::is_trivially_constructible_v<T> &&
    std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T> &&
    std::is_aggregate_v<T>;

/// @}

} // namespace std

#endif // KITSUNE_SUPPORT_TYPE_TRAITS_H
