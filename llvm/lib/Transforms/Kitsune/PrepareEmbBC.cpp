//=- PrepareEmbBC.cpp - Prepare the embedded bitcode for codegen -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prepare the embedded bitcode for code generation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Kitsune/PrepareEmbBC.h"
#include "llvm/Analysis/TapirTargetAnalysis.h"

#define DEBUG_TYPE "prepare-emb-bc"

using namespace llvm;

namespace {

class PrepareEmbBCCuda {
public:
  PrepareEmbBCCuda(const TapirTargetOptions &tto) {}

  bool run(Module &m) {
    // Nothing specifically needs to be done to prepare the module for NVIDIA
    // GPU code generation.
    return false;
  }
};

class PrepareEmbBCHip {
public:
  PrepareEmbBCHip(const TapirTargetOptions &tto) {}

  bool run(Module &m) {
    // TODO: Implement this.
    return false;
  }
};

} // namespace

namespace llvm {

bool PrepareEmbBCPass::run(TapirTargetID tt, Module &m, Module &hostM,
                           ModuleAnalysisManager &hostMAM) {
  const TapirTargetInfo &tgi = hostMAM.getResult<TapirTargetAnalysis>(hostM);
  const TapirTargetOptions &tto = tgi.getOptions();
  switch (tt) {
  case TapirTargetID::Cuda:
    return PrepareEmbBCCuda(tto).run(m);
  case TapirTargetID::Hip:
    return PrepareEmbBCHip(tto).run(m);
  default:
    llvm_unreachable("PrepareEmbBCPass::run: TapirTargetID not handled");
  }
}

} // namespace llvm
