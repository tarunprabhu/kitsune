//===- LowerMobileIntrinsics.h - Lower kitsune mobile intrinsics -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers the Kitsune mobile intrinsics. For now, these are only the
// allocation and deallocation intrinsics, but these may be expanded to include
// explicit memory movement intrinsics as well.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_KITSUNE_LOWER_MOBILE_INTRINSICS_H
#define LLVM_TRANSFORMS_KITSUNE_LOWER_MOBILE_INTRINSICS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// This pass is responsible for lowering the Kitsune mobile intrinsics.
/// This is a module pass because it may require inter-procedural analysis to
/// determine how an intrinsic should be lowered. For instance, the return value
/// from a mobile.alloc intrinsics may be used in a different function inside a
/// loop annotated with a Cuda target. In this case, even if the "main" Tapir
/// target is OpenCilk, we may want to lower the intrinsic to a call to the
/// Cuda UVM allocator.
class LowerMobileIntrinsicsPass
    : public PassInfoMixin<LowerMobileIntrinsicsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  /// This pass is required because the Kitsune intrinsics must be lowered even
  /// if optimizations have been disabled. There is no other way to lower the
  /// intrinsics in the backend.
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_KITSUNE_LOWER_MOBILE_INTRINSICS_H
