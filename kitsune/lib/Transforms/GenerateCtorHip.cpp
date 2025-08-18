//===- GenerateCtorHip.cpp - ctor for Kitsune's hip runtime ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generate global constructors for Kitsune's hip runtime
//
//===----------------------------------------------------------------------===//

#include "GenerateCtorsImpl.h"
#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/Config/config.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbBitcodeUtils.h"
#include "kitsune/Core/EmbDeviceCodeUtils.h"
#include "kitsune/Core/Tapir.h"
#include "kitsune/Core/TapirTargetOptions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace {

/// Helper class to generate a ctor for kithip (Kitsune's runtime for the hip
/// tapir target).
class GenerateCtorHip {
private:
  detail::GetTLI getTLI;
  const TapirTargetOptions &tto;
  const detail::GenerateCtorOptions &genCtorOpts;

private:
  Function *createCtor(Module &m, const Module &devM) {
    const DataLayout &dl = m.getDataLayout();
    LLVMContext &ctx = m.getContext();

    Type *voidTy = Type::getVoidTy(ctx);
    Type *i32 = Type::getInt32Ty(ctx);
    Type *i8 = Type::getInt8Ty(ctx);
    PointerType *ptr = PointerType::getUnqual(ctx);

    FunctionType *fty = FunctionType::get(voidTy, ptr, false);
    Function *ctor =
        Function::Create(fty, GlobalValue::InternalLinkage, ".kithip.ctor", &m);
    TargetLibraryInfo &tli = getTLI(*ctor);
    Type *sizeTTy = tli.getSizeTType(m);

    ConstantInt *ctt = createConstInt(TTID::Hip, ctx);

    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", ctor));

    FunctionCallee kitrtInitialize =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_initialize);
    builder.CreateCall(kitrtInitialize, {ctt});

    // Enable verbose mode early in the constructor so all verbose statements
    // are printed after the runtime has been initialized.
    FunctionCallee kitrtEnableVerbose =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_enable_verbose);
    builder.CreateCall(kitrtEnableVerbose,
                       {ConstantInt::get(i8, tto.getKitrtVerbose(), false)});

    if (tto.getHipXnack() == MaybeBool::On)
      LLVM_DEBUG(dbgs() << "\t\tenable xnack via ctor runtime call.\n");

    FunctionCallee kitrtEnableXnack =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_enable_xnack);
    builder.CreateCall(
        kitrtEnableXnack,
        {ConstantInt::get(i8, tto.getHipXnack() == MaybeBool::On)});

    if (genCtorOpts.useYLaunch)
      LLVM_DEBUG(
          dbgs()
          << "\t\tenable y-axis launch pattern via ctor runtime call.\n");
    FunctionCallee kitrtEnableYAxisLaunch = Intrinsic::getOrInsertDeclaration(
        &m, Intrinsic::kit_enable_y_axis_launches);
    builder.CreateCall(kitrtEnableYAxisLaunch,
                       {ctt, ConstantInt::get(i8, genCtorOpts.useYLaunch)});

    if (unsigned fixedTPB = tto.getFixedThreadsPerBlock()) {
      FunctionCallee kitrtSetFixedTPB =
          Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_set_fixed_tpb);
      builder.CreateCall(kitrtSetFixedTPB,
                         {ctt, ConstantInt::get(i32, fixedTPB)});
    }

    // If the MaxThreadsPerBlock has not been set, use a value of 1024 anyway.
    // At the time of writing, exceeding this value degrades performance. This
    // might change, and we may even have to set a different value depending
    // on the specific GPU architecture.
    //
    // FIXME: Don't hardcode this value here. Maybe move it to a named constant.
    FunctionCallee kitrtSetMaxTPB =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_set_max_tpb);
    unsigned maxTPB = tto.getMaxThreadsPerBlock();
    if (!maxTPB)
      maxTPB = 1024;
    builder.CreateCall(kitrtSetMaxTPB, {ctt, ConstantInt::get(i32, maxTPB)});

    FunctionCallee kitRegisterFatbin =
        getOrInsertLibFunc(&m, tli, LibFunc_kithip_register_fatbin);
    (void)builder.CreateCall(kitRegisterFatbin, {});

    // Register the non-constant global variables reachable from the kernel
    // module. Each of these should have a corresponding global in the host.
    FunctionCallee kitRegisterVar =
        getOrInsertLibFunc(&m, tli, LibFunc_kithip_register_var);
    for (const GlobalVariable &devG : devM.globals()) {
      if (devG.isConstant())
        continue;

      GlobalVariable *hostG = m.getGlobalVariable(devG.getName());
      assert(hostG && "Could not find corresponding global on host");
      LLVM_DEBUG(dbgs() << "\t\t\tregister global '" << hostG->getName()
                        << "' via ctor runtime call.\n");

      StringRef name = hostG->getName();
      Type *type = hostG->getValueType();

      GlobalVariable *gName = createConstString(name, m);
      Constant *gSize = ConstantInt::get(sizeTTy, dl.getTypeAllocSize(type));
      Value *args[] = {hostG, gName, gSize};
      (void)builder.CreateCall(kitRegisterVar, args);
    }

    builder.CreateRetVoid();
    return ctor;
  }

public:
  GenerateCtorHip(detail::GetTLI getTLI, const TapirTargetOptions &tto,
                  const detail::GenerateCtorOptions &genCtorOpts)
      : getTLI(getTLI), tto(tto), genCtorOpts(genCtorOpts) {}

  void run(Module &m) {
    // Set the priority of this ctor to be very low so it is one of the last to
    // run.
    std::unique_ptr<Module> devM = getEmbModule(TTID::Hip, m);
    Function *ctor = createCtor(m, *devM);
    appendToGlobalCtors(m, ctor, 65536);
  }
};

} // namespace

void llvm::detail::genCtorHip(Module &m, detail::GetTLI getTLI,
                              const TapirTargetOptions &tto,
                              const detail::GenerateCtorOptions &genCtorOpts) {
  GenerateCtorHip(getTLI, tto, genCtorOpts).run(m);
}
