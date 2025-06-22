//===- EmbBCPassUtils.cpp - Utilities for embedded bitcode passes ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities shared by embedded bitcode passes.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EmbBCPassUtils.h"
#include "kitsune/Core/TapirTargetOptions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/OptimizationLevel.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;

static std::unique_ptr<Module> parseLibDeviceBCFile(StringRef file,
                                                    LLVMContext &ctx) {
  SMDiagnostic sm;
  std::unique_ptr<Module> m = parseIRFile(file, sm, ctx);
  if (not m)
    report_fatal_error(StringRef(join_items(
        "", "Failed to parse libdevice bitcode file: ", sm.getMessage())));
  return m;
}

static std::unique_ptr<Module>
getLibDeviceModuleCuda(const TapirTargetOptions &tto, LLVMContext &ctx) {
  return parseLibDeviceBCFile(tto.getCudaRuntimeBCFile(), ctx);
}

static std::unique_ptr<Module>
getLibDeviceModuleHip(const TapirTargetOptions &tto, LLVMContext &ctx) {
  const std::vector<std::string> &bcFiles = tto.getHipRuntimeBCFiles();
  std::unique_ptr<Module> libDeviceM = parseLibDeviceBCFile(bcFiles[0], ctx);

  Linker linker(*libDeviceM);
  for (size_t i = 1; i < bcFiles.size(); ++i) {
    StringRef bcFile = bcFiles[i];
    std::unique_ptr<Module> m = parseLibDeviceBCFile(bcFile, ctx);
    if (linker.linkInModule(std::move(m), Linker::OverrideFromSrc))
      report_fatal_error("Error linking device bitcode file: " + bcFile);
  }
  return libDeviceM;
}

std::unique_ptr<Module> llvm::getLibDeviceModule(TTID tt,
                                                 const TapirTargetOptions &tto,
                                                 LLVMContext &ctx) {
  switch (tt) {
  case TTID::Cuda:
    return getLibDeviceModuleCuda(tto, ctx);
  case TTID::Hip:
    return getLibDeviceModuleHip(tto, ctx);
  default:
    llvm_unreachable("getLibDeviceBCFile: TTID not handled");
  }
}
