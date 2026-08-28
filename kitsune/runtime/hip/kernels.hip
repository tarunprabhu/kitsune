//===- hip/kernels.cpp - GPU kernels for Kitsune's hip runtime ------------===//
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

#include "kernels.h"

#include <hip/hip_runtime.h>

namespace kitrt {

__global__ void kithipMemset64Kernel(uint64_t *buf, uint64_t n, uint64_t init) {
  uint64_t i =
      (uint64_t)blockDim.x * (uint64_t)blockIdx.x + (uint64_t)threadIdx.x;
  if (i < n)
    buf[i] = init;
}

} // namespace kitrt

void kitrt::kithipMemset64Launch(uint64_t *buf, uint64_t n, uint64_t init) {
  // TODO: check if a different value of threads-per-block performs better.
  constexpr unsigned tpb = 128;
  unsigned bpg = (n + tpb - 1) / tpb;
  kitrt::kithipMemset64Kernel<<<bpg, tpb>>>(buf, n, init);
}
