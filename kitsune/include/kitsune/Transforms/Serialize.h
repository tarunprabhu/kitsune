//===- Serialize.h - Serialize certain tapir constructs ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Serialize certain tapir constructs.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_SERIALIZE_H
#define KITSUNE_TRANSFORMS_SERIALIZE_H

#include "kitsune/Passes/DependentPass.h"
#include "kitsune/Transforms/AnnotateTapirLoops.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Serialize certain tapir constructs.
class SerializePass
    : public DependentPass<SerializePass, AnnotateTapirLoopsPass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_SERIALIZE_H
