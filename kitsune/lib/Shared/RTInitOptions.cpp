//===- RTInitOptions.cpp - Options to initialize Kitsune's runtime --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Initialization options for Kitsune's runtime.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Shared/RTInitOptions.h"
#include "kitsune/Support/TypeTraits.h"

// kitsune/Shared/RTInitOptions.h is intended to be C-safe, so we add this
// assertion here.

static_assert(std::is_interop_v<kitrt::InitOptions> &&
              "kitrt::InitOptions must be a trivial type with the standard "
              "memory layout");
