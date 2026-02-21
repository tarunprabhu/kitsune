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

#include "llvm/IR/PassManager.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

/// This pass computes properties of tapir loops and adds them to the loop
/// annotations. These can be read by passes that run later in the
/// pipeline. These are intended to inform how the loop will be transformed
/// prior to loop spawning and may also be used to affect how the loop will be
/// spawned. In some cases, subsequent passes may even serialize tapir loops.
///
/// For example, a tapir loop that performs a reduction will be annotated with
/// the tapir.loop.reduction. If compiling for a GPU, a pass that runs before
/// loop spawning will examine this annotation and transform the loop to a form
/// suitable for computing parallel reductions on a GPU. That pass will ignore
/// loops that do not contain the attribute.
///
class AnnotateTapirLoopsPass : public PassInfoMixin<AnnotateTapirLoopsPass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);

  /// This pass is required because passes that run later in the pipeline may
  /// not work correctly if the annotations are not computed.
  static bool isRequired() { return true; }
};

/// @}

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_ANNOTATE_TAPIR_LOOPS_H
