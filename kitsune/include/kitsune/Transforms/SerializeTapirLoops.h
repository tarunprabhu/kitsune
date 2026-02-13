//===- SerializeTapirLoops.h - Serialize certain tapir loops ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to analyze tapir loop nests and serialize certain tapir loops.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_SERIALIZE_TAPIR_LOOPS_H
#define KITSUNE_TRANSFORMS_SERIALIZE_TAPIR_LOOPS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// Pass that analyzes tapir loop nests and serializes any tapir loops that
/// either cannot be lowered using a tapir target, or that may degrade
/// performance if lowered using a tapir target. This pass expects the
/// annotate-tapir-loops pass to have been run.
///
class SerializeTapirLoopsPass : public PassInfoMixin<SerializeTapirLoopsPass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_SERIALIZE_TAPIR_LOOPS_H
