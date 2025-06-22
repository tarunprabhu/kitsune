//===- TargetUtils.cpp - Helper functions for targets/machines ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for targets and target machines.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TargetUtils.h"
#include "kitsune/Core/TapirTargetOptions.h"
#include "kitsune/Support/OptLevelUtils.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"

using namespace llvm;

static TargetMachine *createAMDGPUTargetMachine(const TapirTargetOptions &tto) {
  Triple triple("amdgcn", "amd", "amdhsa");

  std::string err;
  const Target *target = TargetRegistry::lookupTarget("", triple, err);
  assert(target && "Unable to find registered AMDGPU target");

  // TODO: Should we allow relocations here?
  CodeModel::Model codeModel = CodeModel::Small;
  Reloc::Model relocModel = Reloc::Static;
  CodeGenOptLevel optLevel = mapToCodeGenOptLevel(tto.getOptLevel());
  TargetOptions opts;
  opts.UseInitArray = true;
  opts.EmitAddrsig = true;
  opts.AllowFPOpFusion = tto.getFPOpFusionMode();

  return target->createTargetMachine(triple.str(), tto.getHipArch(),
                                     tto.getHipTargetFeatures(), opts,
                                     relocModel, codeModel, optLevel);
}

static TargetMachine *createPTXTargetMachine(const TapirTargetOptions &tto) {
  Triple triple("nvptx64", "nvidia", "cuda");

  std::string err;
  const Target *target = TargetRegistry::lookupTarget("", triple, err);
  assert(target && "Unable to find registered PTX target");

  CodeModel::Model codeModel = CodeModel::Small;
  Reloc::Model relocModel = Reloc::PIC_;
  CodeGenOptLevel optLevel = mapToCodeGenOptLevel(tto.getOptLevel());
  TargetOptions opts;
  opts.AllowFPOpFusion = tto.getFPOpFusionMode();

  return target->createTargetMachine(triple.str(), tto.getCudaArch(),
                                     tto.getCudaTargetFeatures(), opts,
                                     relocModel, codeModel, optLevel);
}

TargetMachine *llvm::createTargetMachine(TTID tt,
                                         const TapirTargetOptions &tto) {
  switch (tt) {
  case TTID::Cuda:
    return createPTXTargetMachine(tto);
  case TTID::Hip:
    return createAMDGPUTargetMachine(tto);
  default:
    llvm_unreachable("createTargetMachine: TTID not handled");
  }
}
