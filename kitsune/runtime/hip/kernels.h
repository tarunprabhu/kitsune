//===- hip/kernels.h - GPU kernels for Kitsune's hip runtime ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// GPU kernels for Kitsune's hip runtime.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_HIP_KERNELS_H
#define KITRT_HIP_KERNELS_H

#include <cstdint>

namespace kitrt {

/// Launch a kernel that implements a memset for 64-bit values.
///
/// \param buf The buffer to initialize
/// \param n The number of elements to set
/// \param init The value to write  to each element
void kithipMemset64Launch(uint64_t *buf, uint64_t n, uint64_t init);

} // namespace kitrt

#endif // KITRT_HIP_KERNELS_H
