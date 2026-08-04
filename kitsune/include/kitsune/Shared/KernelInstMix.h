//===- KernelInstMix.h - Kernel properties for the runtime ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for kernel properties determined at compile-time and used at runtime.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SHARED_KERNELINSTMIX_H
#define KITSUNE_SHARED_KERNELINSTMIX_H

#include <stdint.h>
#include <type_traits>

/// \addtogroup kitsune
/// @{

namespace kitrt {

/// *** EXPERIMENTAL:
///
/// This contains information about the instruction kinds in a kernel and the
/// number of device-side functions called from it. It is gathered from the LLVM
/// form of the code (not ptx/s-code) and at this point is limited. In general
/// we are using to explore impacts on launch parameters.
struct KernelInstMix {
public:
  uint64_t memOps;
  uint64_t fpOps;
  uint64_t intOps;
  uint64_t otherOps;

public:
  KernelInstMix() = default;

  KernelInstMix &operator+=(const KernelInstMix &o) {
    this->memOps += o.memOps;
    this->fpOps += o.fpOps;
    this->intOps += o.intOps;
    this->otherOps += o.otherOps;

    return *this;
  }
};

static_assert(
    std::is_trivially_constructible_v<KernelInstMix> &&
    std::is_trivially_copyable_v<KernelInstMix> &&
    std::is_standard_layout_v<KernelInstMix> &&
    "KernelInstMix must be a trivial type with the standard memory layout");

} // namespace kitrt

/// @}

#endif // KITSUNE_SHARED_KERNELINSTMIX_H
