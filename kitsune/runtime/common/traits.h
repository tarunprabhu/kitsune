//===- traits.h - Type traits -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type traits
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_COMMON_TRAITS_H
#define KITRT_COMMON_TRAITS_H

#include <type_traits>

// We add the traits to std for consistency with other type traits.
namespace std {

namespace detail {

// Trait to check if a type is complete.
template <typename T, typename = void> struct is_complete : std::false_type {};
template <typename T>
struct is_complete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};

} // namespace detail

/// Check if a given type is complete, i.e. it is not a forward declaration.
template <typename T>
static constexpr bool is_complete_v = detail::is_complete<T>::value;

} // namespace std

#endif // KITRT_COMMON_TRAITS_H
