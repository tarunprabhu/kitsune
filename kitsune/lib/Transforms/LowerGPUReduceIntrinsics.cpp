//===- LowerGPUReduceIntrinsics.cpp - Lower GPU reduce intrinsics ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune-specific GPU reduction intrinsics.
//
//===----------------------------------------------------------------------===//

#include "LowerGPUIntrinsicsImpl.h"

using namespace llvm;

bool detail::LowerGPUIntrImpl::lowerReduceDirectIntr(CallInst *call) {
  // TODO: Implement this.
  return true;
}

bool detail::LowerGPUIntrImpl::lowerReduceShmemIntr(CallInst *call) {
  // TODO: Implement this.
  return true;
}

bool detail::LowerGPUIntrImpl::lowerReduceWarpShflIntr(CallInst *call) {
  // TODO: Implement this.
  return true;
}

bool detail::LowerGPUIntrImpl::lowerReduceWarpShflShmemIntr(CallInst *call) {
  // TODO: Implement this.
  return true;
}
