//===- TTUtils.cpp - Utilities closely related to tapir targets -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Miscellaneous utilities for tapir targets and TTID's.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TTUtils.h"
#include "kitsune/Config/Config.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/TTOptions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;

static Expected<OwnedModule> parseLLVMFile(StringRef file, LLVMContext &ctx) {
  SMDiagnostic diag;
  if (OwnedModule m = parseIRFile(file, diag, ctx))
    return m;
  return createDiagError(DiagID::ErrParseLLVM, diag.getMessage());
}

static Expected<OwnedModule> getLibDeviceModuleCuda(const TTOptions &tto,
                                                    LLVMContext &ctx) {
  return parseLLVMFile(tto.getCudaRuntimeBCFile(), ctx);
}

static Expected<OwnedModule> getLibDeviceModuleHip(const TTOptions &tto,
                                                   LLVMContext &ctx) {
  const std::vector<std::string> &bcFiles = tto.getHipRuntimeBCFiles();
  assert(tto.getHipRuntimeBCFiles().size() &&
         "At least one bitcode file is required by Hip's libDevice module");

  Expected<OwnedModule> libDeviceM = parseLLVMFile(bcFiles[0], ctx);
  if (!libDeviceM)
    return libDeviceM.takeError();
  Linker linker(**libDeviceM);
  for (size_t i = 1; i < bcFiles.size(); ++i) {
    StringRef bcFile = bcFiles[i];
    Expected<OwnedModule> m = parseLLVMFile(bcFile, ctx);
    if (!m)
      return m;
    if (linker.linkInModule(std::move(*m), Linker::OverrideFromSrc))
      return createDiagError(DiagID::ErrLinkLLVM, bcFile);
  }
  return libDeviceM;
}

static Expected<OwnedModule> getRuntimeModuleOpenCilk(const TTOptions &tto,
                                                      LLVMContext &ctx) {
  return parseLLVMFile(tto.getOpenCilkRuntimeBCFile(), ctx);
}

Expected<OwnedModule> llvm::getSupportModule(TTID tt, const TTOptions &tto,
                                             LLVMContext &ctx) {
  switch (tt) {
  case TTID::Cuda:
    return getLibDeviceModuleCuda(tto, ctx);
  case TTID::Hip:
    return getLibDeviceModuleHip(tto, ctx);
  case TTID::OpenCilk:
    return getRuntimeModuleOpenCilk(tto, ctx);
  default:
    llvm_unreachable("getSupportModule: TTID not handled");
  }
}

Expected<OwnedModule> llvm::getLibDeviceModule(TTID tt, const TTOptions &tto,
                                               LLVMContext &ctx) {
  switch (tt) {
  case TTID::Cuda:
  case TTID::Hip:
    return getSupportModule(tt, tto, ctx);
  default:
    llvm_unreachable("getLibDeviceModule: TTID not handled");
  }
}

TapirSpawnStrategy llvm::getSpawnStrategyFor(TTID tt) {
  switch (tt) {
  case llvm::TTID::Nolo:
  case llvm::TTID::Serial:
    return llvm::TapirSpawnStrategy::Sequential;
  case llvm::TTID::Cuda:
  case llvm::TTID::Hip:
    return llvm::TapirSpawnStrategy::GPU;
  case llvm::TTID::OpenCilk:
    return llvm::TapirSpawnStrategy::DivideAndConquer;
  case llvm::TTID::Custom:
  case llvm::TTID::OpenMP:
  case llvm::TTID::Pthreads:
  case llvm::TTID::Qthreads:
    return llvm::TapirSpawnStrategy::Basic;
  default:
    break;
  }
  llvm_unreachable("getSpawnStrategyFor: TTID not handled");
}

static bool contains(ArrayRef<TTID> tts, TTID key) {
  for (TTID tt : tts)
    if (tt == key)
      return true;
  return false;
}

bool llvm::isGPUTT(TTID tt) { return contains(kitKnownGPUTTs(), tt); }

bool llvm::isCPUTT(TTID tt) { return contains(kitKnownCPUTTs(), tt); }

bool llvm::generatesEmbBC(TTID tt) { return contains(kitKnownEmbBCTTs(), tt); }

bool llvm::isEnabledTT(TTID tt) {
  // TTID::Nolo will never be in the known tapir target list since it is not a
  // "true" tapir target. Nevertheless, it must be treated as "always enabled".
  if (tt == TTID::Nolo)
    return true;
  return contains(kitEnabledTTs(), tt);
}
