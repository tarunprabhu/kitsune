//===- FunctionUtils.h - Utilities for LLVM functions ----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM Function's.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_FUNCTION_UTILS_H
#define KITSUNE_CORE_FUNCTION_UTILS_H

namespace llvm {

/// \addtogroup kitsune
/// @{

class Argument;
class Function;

/// Copy function attributes and other properties from the function \p src to
/// the function \p dst. This will *NOT* copy attributes on function arguments.
/// In order to copy those, use the copyAttributesFrom() method on a Function.
///
///  - calling convention
///  - garbage collection algorithm
///  - personality function
///  - prefix data
///  - prolog data
///
void copyAttrs(Function &dst, Function &src);

/// Copy attributes from the argument \p src to the argument \p dst.
void copyAttrs(Argument &dst, Argument &src);

/// @}

} // namespace llvm

#endif // KITSUNE_CORE_FUNCTION_UTILS_H
