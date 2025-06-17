//=- LinkDeviceBitcode.cpp - Link device bitcode into the embedded bitcode --=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Link the device bitcode file(s) into the embedded bitcode modules.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Kitsune/LinkDeviceBitcode.h"
#include "llvm/Analysis/TapirTargetAnalysis.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/Kitsune/EmbBCPassUtils.h"

#define DEBUG_TYPE "link-device-bitcode"

using namespace llvm;

namespace llvm {

bool LinkDeviceBitcodePass::run(TapirTargetID tt, Module &devM, Module &hostM,
                                ModuleAnalysisManager &hostMAM) {
  LLVMContext &ctx = devM.getContext();
  const TapirTargetInfo &tgi = hostMAM.getResult<TapirTargetAnalysis>(hostM);
  const TapirTargetOptions &tto = tgi.getOptions();

  Linker linker(devM);
  std::unique_ptr<Module> libDeviceM = getLibDeviceModule(tt, tto, ctx);
  if (linker.linkInModule(std::move(libDeviceM), Linker::LinkOnlyNeeded))
    report_fatal_error("Error linking device module");

  return true;
}

} // namespace llvm
