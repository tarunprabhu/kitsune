//===- CheckTypes.cpp - Assertions on shared types ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Assertions on shared types.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Shared/RTInitOptions.h"
#include "kitsune/Shared/TypeTraits.h"

using namespace kitrt;

// The headers in kitsune/Shared are intended to be C-safe. While we could wrap
// the includes of kitsune/Shared/TypeTraits.h in #ifdef __cplusplus, adding
// all the static assertions here instead is, arguably, cleaner.

static_assert(RTID::RT_COMMON == 0 && "Value of RT_COMMON must be 0");

static_assert(
    std::is_interop_v<InitOptions> &&
    "InitOptions must be a trivial type with the standard memory layout");
