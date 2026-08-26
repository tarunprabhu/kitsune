//===- LowerKitReduceIntrinsics.h - Lowering reduce intrinsics -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune's reduce intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_LOWER_KIT_REDUCE_INTRINSICS_H
#define KITSUNE_TRANSFORMS_LOWER_KIT_REDUCE_INTRINSICS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Lower Kitsune's reduce intrinsics.
class LowerKitReduceIntrinsicsPass
    : public PassInfoMixin<LowerKitReduceIntrinsicsPass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);
};

} // end namespace llvm

#endif // KITSUNE_TRANSFORMS_LOWER_KIT_REDUCE_INTRINSICS_H
