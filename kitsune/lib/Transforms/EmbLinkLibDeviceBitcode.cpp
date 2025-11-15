//===- EmbLinkLibDeviceBitcode.cpp - Link libdevice and embedded modules --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Link the libdevice bitcode module(s) into the embedded bitcode modules.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EmbLinkLibDeviceBitcode.h"
#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/Core/TTUtils.h"
#include "llvm/Linker/Linker.h"

#define DEBUG_TYPE "emb-link-libdevice-bitcode"

using namespace llvm;

namespace llvm {

bool EmbLinkLibDeviceBitcodePass::run(TTID tt, Module &devM, Module &hostM,
                                      ModuleAnalysisManager &hostMAM) {
  LLVMContext &ctx = devM.getContext();
  const TapirTargetInfo &tgi = hostMAM.getResult<TapirTargetAnalysis>(hostM);
  const TTOptions &tto = tgi.getOptions();

  Linker linker(devM);
  std::unique_ptr<Module> libDeviceM = getLibDeviceModule(tt, tto, ctx);
  if (linker.linkInModule(std::move(libDeviceM), Linker::LinkOnlyNeeded))
    report_fatal_error("Error linking device module");

  return true;
}

} // namespace llvm
