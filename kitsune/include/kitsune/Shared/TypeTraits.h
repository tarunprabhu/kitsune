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

} // namespace std

#endif // KITRT_COMMON_TRAITS_H
