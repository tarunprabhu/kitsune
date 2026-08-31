//===- LowerGPUWarpIntrinsics.cpp - Lower GPU warp intrinsics -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune-specific GPU warp intrinsics.
//
//===----------------------------------------------------------------------===//

#include "LowerGPUIntrinsicsImpl.h"

using namespace llvm;

bool llvm::detail::lowerGPUWarpShflDownSyncIntr(CallInst *call) {
  // TODO: Implement this.
  return true;
}

bool llvm::detail::lowerGPUWarpIdIntr(CallInst *call) {
  // TODO: Implement this.
  return true;
}

bool llvm::detail::lowerGPUWarpLaneIntr(CallInst *call) {
  // TODO: Implement this.
  return true;
}

bool llvm::detail::lowerGPUWarpSizeIntr(CallInst *call) {
  // TODO: Implement this.
  return true;
}
