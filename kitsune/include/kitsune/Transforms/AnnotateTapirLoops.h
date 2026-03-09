//===- AnnotateTapirLoops.h - Annotate tapir loops --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to analyze tapir loops and add appropriate annotations that will be
// used by subsequent passes.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_ANNOTATE_TAPIR_LOOPS_H
#define KITSUNE_TRANSFORMS_ANNOTATE_TAPIR_LOOPS_H

#include "kitsune/Passes/RequirablePass.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Analyze tapir loops and add annotations that will be used by passes that run
/// later in the pipeline.
class AnnotateTapirLoopsPass : public PassInfoMixin<AnnotateTapirLoopsPass>,
                               public RequirablePass<AnnotateTapirLoopsPass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);

  /// This pass is required because passes that run later in the pipeline may
  /// not work correctly if the annotations are not computed.
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_ANNOTATE_TAPIR_LOOPS_H
