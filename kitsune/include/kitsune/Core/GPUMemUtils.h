//===- GPUMemUtils.h - Utilities for GPU memory -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Types and utilities to work with GPU memory.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_GPU_MEM_UTILS_H
#define KITSUNE_CORE_GPU_MEM_UTILS_H

#include "kitsune/Support/FromInt.h"
#include "kitsune/Support/FromString.h"
#include "kitsune/Support/ToString.h"

#include <cstdint>

namespace llvm {

/// When dynamically allocating memory for use on a GPU, we can either allocate
/// the memory directly in GPU's global memory, or in unified virtual memory.
/// This enum may be used in situations where we have to make a choice between
/// allocating the one or the other.
enum class GPUDynMemAllocKind : uint32_t {
  Global = 1,
  UVM,
};

} // namespace llvm

#endif // KITSUNE_CORE_GPU_MEM_UTILS_H
