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
#include "kitsune/Transforms/Utils/EmbModulePassUtils.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Linker/Linker.h"

#define DEBUG_TYPE "emb-link-libdevice-bitcode"

using namespace llvm;

namespace llvm {

bool EmbLinkLibDeviceBitcodePass::run(TTID tt, Module &devM, Module &hostM,
                                      ModuleAnalysisManager &hostMAM) {
  LLVMContext &ctx = devM.getContext();
  const TapirTargetInfo &tgi = hostMAM.getResult<TapirTargetAnalysis>(hostM);
  const TapirTargetOptions &tto = tgi.getOptions();

  // Collect the functions which are declarations in the current module.
  SmallSet<StringRef, 16> decls;
  for (Function &f : devM.functions())
    if (f.hasName() and not f.size())
      decls.insert(f.getName());

  Linker linker(devM);
  std::unique_ptr<Module> libDeviceM = getLibDeviceModule(tt, tto, ctx);
  if (linker.linkInModule(std::move(libDeviceM), Linker::LinkOnlyNeeded))
    report_fatal_error("Error linking device module");

  // If definitions for declared functions have been provided by the libdevice
  // module, change the linkage of the definitions to linkonce_odr. Without
  // this, we may get link-time errors if multiple translation units have linked
  // the same libdevice function into the embedded module.
  for (Function &f : devM.functions())
    if (f.hasName() and f.size())
      if (decls.contains(f.getName()))
        f.setLinkage(GlobalValue::LinkOnceODRLinkage);

  return true;
}

} // namespace llvm
