//===- EmbBCPassUtils.cpp - Utilities for embedded bitcode passes ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Utilities shared by embedded bitcode passes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Kitsune/EmbBCPassUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Frontend/Tapir/OptLevelUtils.h"
#include "llvm/Frontend/Tapir/TapirTargetOptions.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
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

std::unique_ptr<Module> llvm::getLibDeviceModule(TapirTargetID tt,
                                                 const TapirTargetOptions &tto,
                                                 LLVMContext &ctx) {
  switch (tt) {
  case TapirTargetID::Cuda:
    return getLibDeviceModuleCuda(tto, ctx);
  case TapirTargetID::Hip:
    return getLibDeviceModuleHip(tto, ctx);
  default:
    llvm_unreachable("getLibDeviceBCFile: TapirTargetID not handled");
  }
}
