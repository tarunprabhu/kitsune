//===- CreateEmbBitcode.h - Create an embedded bitcode global --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Create an embedded module to create embedded bitcode. Clone the device
// functions into it.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_CREATE_EMB_BITCODE_H
#define KITSUNE_TRANSFORMS_CREATE_EMB_BITCODE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// Create an embedded bitcode module for each of the tapir targets that are to
/// be used. A corresponding (uninitialized) embedded fat binary global variable
/// is also created since the verifier requires that a fat binary global be
/// present for every embedded bitcode global. All functions in the host module
/// with the kit_device attribute are cloned into the embedded module, together
/// with any functions and other global values that are reachable from these
/// attributed functions.
///
/// However, the global values reachable from tapir loops are not cloned into
/// the created module.
///
/// FIXME: The reason for not handling the reachable global values from tapir
/// loops here is that it is not yet clear exactly where in the pass pipeline
/// this pass will be inserted. For now, it is part of Kitsune's pre-tapir
/// pipeline, but it is may be worthwhile moving this earlier in the pipeline.
/// In that case, we may not be able to identify tapir loops, and therefore will
/// not be able to collect reachable GlobalValue's. If we decide to keep this
/// pass in the pre-tapir pipeline, it may be worthwhile computing and cloning
/// the reachable GlobalValue's here.
class CreateEmbBitcodePass : public PassInfoMixin<CreateEmbBitcodePass> {
public:
  PreservedAnalyses run(Module &hostM, ModuleAnalysisManager &mam);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_CREATE_EMB_BITCODE_H
