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

namespace {

class LinkDeviceBitcodeCuda {
private:
  const TapirTargetOptions &tto;

public:
  LinkDeviceBitcodeCuda(const TapirTargetOptions &tto) : tto(tto) {}

  bool run(Module &m) {
    StringRef file = tto.getCudaRuntimeBCFile();
    LLVMContext &ctx = m.getContext();
    std::unique_ptr<Module> libDeviceM = parseLibDeviceBCFile(file, ctx);

    Linker linker(m);
    linker.linkInModule(std::move(libDeviceM), Linker::LinkOnlyNeeded);
    return true;
  }
};

class LinkDeviceBitcodeHip {
private:
  const TapirTargetOptions &tto;

public:
  LinkDeviceBitcodeHip(const TapirTargetOptions &tto) : tto(tto) {}

  bool run(Module &m) {
    // TODO: Implement this.
    return false;
  }
};

} // namespace

namespace llvm {

bool LinkDeviceBitcodePass::run(TapirTargetID tt, Module &m, Module &hostM,
                                ModuleAnalysisManager &hostMAM) {
  const TapirTargetInfo &tgi = hostMAM.getResult<TapirTargetAnalysis>(hostM);
  const TapirTargetOptions &tto = tgi.getOptions();
  switch (tt) {
  case TapirTargetID::Cuda:
    return LinkDeviceBitcodeCuda(tto).run(m);
  case TapirTargetID::Hip:
    return LinkDeviceBitcodeHip(tto).run(m);
  default:
    llvm_unreachable("LinkDeviceBitcodePass: TapirTargetID not handled");
  }
}

} // namespace llvm
