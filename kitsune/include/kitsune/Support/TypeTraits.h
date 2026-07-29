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

#include <type_traits>

// These are added to the std namespace for consistency with other standard
// traits.
namespace std {

/// \addtogroup kitsune
/// @{

template <typename T, typename = void>
struct is_iterable_t : std::false_type {};

template <typename T>
struct is_iterable_t<T, std::void_t<decltype(std::declval<T>().begin()),
                                    decltype(std::declval<T>().end())>>
    : std::true_type {};

/// Return true if the type is iterable. This will be the case if the type has
/// begin() and end() members. This makes the type suitable to be used in a
/// range-based for loop.
template <typename T>
static constexpr bool is_iterable_v = is_iterable_t<T>::value;

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
/// as we as the cv-qualified variants of these. This will not return true if
/// `T` is a reference type.
template <typename T>
static constexpr bool is_float_v = std::is_same_v<std::remove_cv_t<T>, float>;

/// Check if the type is a 64-bit floating point type. This will return on the
/// following types and their aliases:
///
///     double
///
/// as we as the cv-qualified variants of these. This will not return true if
/// `T` is a reference type.
template <typename T>
static constexpr bool is_double_v = std::is_same_v<std::remove_cv_t<T>, double>;

/// Check if the type is a long double type. This will return on the following
/// types and their aliases:
///
///     long double
///
/// as we as the cv-qualified variants of these. This will not return true if
/// `T` is a reference type.
template <typename T>
static constexpr bool is_long_double_v =
    std::is_same_v<std::remove_cv_t<T>, long double>;

/// @}

} // namespace std

#endif // KITSUNE_SUPPORT_TYPE_TRAITS_H
