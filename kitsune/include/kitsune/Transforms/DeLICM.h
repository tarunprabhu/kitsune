//===- DeLICM.h - Pass that is the inverse of LICM --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass that is the inverse of the LICM pass.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_DELICM_H
#define KITSUNE_TRANSFORMS_DELICM_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Pass that is the inverse of the LICM pass. Loop-invariant instructions may
/// be sunk into loops from which they may have been hoisted.
class DeLICMPass : public PassInfoMixin<DeLICMPass> {
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &am);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_DELICM_H
