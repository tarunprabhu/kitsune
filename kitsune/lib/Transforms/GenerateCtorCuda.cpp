//===- GenerateCtorCuda.cpp - ctor for Kitsune's cuda runtime -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generate global constructors for Kitsune's cuda runtime
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

/// Helper class to generate a ctor for kitcuda (Kitsune's runtime for the cuda
/// tapir target). This will also create variables for the fat binary and the
/// bundle containing the fat binary that is needed by the cuda runtime.
///
/// Registering the fat binary image (and all the associated components) is
/// an undocumented portion of the CUDA API. One place to peek for some details
/// hides in the cuda header files; specifially fatbinary_section.h. This shows
/// the following struct that we need to have in the host side code.
///
///    struct fatbinC_Wrapper_t {
///      int magic;
///      int version;
///      const unsigned long long *data;
///      void *filename_or_fatbins;
///    };
///
/// * Per the header, the magic number is 0x466243B1
/// * FATBINC_VERSION is 1 and FATBINC_LINK_VERSION is 2 (more below)
/// * Then section and segments are needed that contains the "fatbin control
///   structure".  This loosely looks like:
///
///        Control section name: ".nvFatBinSegment"
///        Fatbinary section name: ".nv_fatbin"
///        Pre-linked relocatable section: "__nv_relfatbin"
///
/// * The last struct member varies between versions. In the case of version 1
///   it can be a offline filename and for version 2 it is an array of
///   pre-linked fatbins.
///
class GenerateCtorCuda {
private:
  detail::GetTLI getTLI;
  const TapirTargetOptions &tto;
  const detail::GenerateCtorOptions &genCtorOpts;

private:
  Function *createCtor(Module &m, const Module &devM) {
    const DataLayout &dl = m.getDataLayout();

    LLVMContext &ctx = m.getContext();

    Type *voidTy = Type::getVoidTy(ctx);
    PointerType *ptrTy = PointerType::getUnqual(ctx);
    Type *i32Ty = Type::getInt32Ty(ctx);
    Type *boolTy = Type::getInt8Ty(ctx);

    ConstantInt *constTT = createConstInt(TTID::Cuda, ctx);

    FunctionType *ctorTy = FunctionType::get(voidTy, ptrTy, false);
    Function *ctor = Function::Create(ctorTy, GlobalValue::InternalLinkage,
                                      ".kitcuda.ctor", &m);

    TargetLibraryInfo &tli = getTLI(*ctor);
    Type *sizeTTy = tli.getSizeTType(m);

    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", ctor));

    FunctionCallee kitrtInitialize =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_initialize);
    builder.CreateCall(kitrtInitialize, {constTT});

    // Enable verbose mode early in the constructor so all verbose statements
    // are printed after the runtime has been initialized.
    FunctionCallee kitrtEnableVerbose =
        Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_enable_verbose);
    builder.CreateCall(
        kitrtEnableVerbose,
        {ConstantInt::get(boolTy, tto.getKitrtVerbose(), false)});

    if (unsigned fixedTPB = tto.getFixedThreadsPerBlock()) {
      FunctionCallee kitrtSetFixedTPB =
          Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_set_fixed_tpb);
      builder.CreateCall(kitrtSetFixedTPB,
                         {constTT, ConstantInt::get(i32Ty, fixedTPB)});
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
    builder.CreateCall(kitrtSetMaxTPB,
                       {constTT, ConstantInt::get(i32Ty, maxTPB)});

    FunctionCallee kitrtEnableRefineLaunches =
        Intrinsic::getOrInsertDeclaration(
            &m, Intrinsic::kit_enable_refine_launches);
    builder.CreateCall(
        kitrtEnableRefineLaunches,
        {constTT, ConstantInt::get(boolTy, genCtorOpts.refineLaunches)});

    FunctionCallee kitRegisterFatbin =
        getOrInsertLibFunc(&m, tli, LibFunc_kitcuda_register_fatbin);
    (void)builder.CreateCall(kitRegisterFatbin, {});

    // Register the non-constant global variables reachable from the kernel
    // module. Each of these should have a corresponding global in the host.
    FunctionCallee kitRegisterVar =
        getOrInsertLibFunc(&m, tli, LibFunc_kitcuda_register_var);
    for (const GlobalVariable &devG : devM.globals()) {
      if (devG.isConstant())
        continue;

      GlobalVariable *hostG = m.getGlobalVariable(devG.getName());
      assert(hostG && "Could not find corresponding global on host");

      StringRef name = hostG->getName();
      Type *type = hostG->getValueType();

      GlobalVariable *gName = createConstString(name, m);
      Constant *gSize = ConstantInt::get(sizeTTy, dl.getTypeAllocSize(type));
      Value *args[] = {hostG, gName, gSize};
      (void)builder.CreateCall(kitRegisterVar, args);
    }

    // Wrap up fatbinary registration steps.
    FunctionCallee kitRegisterFatbinEnd =
        getOrInsertLibFunc(&m, tli, LibFunc_kitcuda_register_fatbin_end);
    (void)builder.CreateCall(kitRegisterFatbinEnd, {});

    builder.CreateRetVoid();
    return ctor;
  }

public:
  GenerateCtorCuda(detail::GetTLI getTLI, const TapirTargetOptions &tto,
                   const detail::GenerateCtorOptions &genCtorOpts)
      : getTLI(getTLI), tto(tto), genCtorOpts(genCtorOpts) {}

  void run(Module &m) {
    GlobalVariable *gFB = getSingletonFBGlobal(TTID::Cuda, m);
    assert(gFB && "Could not find global with embedded cuda fat binary");

    GlobalVariable *gBC = getEmbBCGlobal(TTID::Cuda, m);
    assert(gBC && "Could not find global with embedded bitcode");

    std::unique_ptr<Module> devM = parseEmbBCGlobal(*gBC);
    Function *ctor = createCtor(m, *devM);

    // Set the priority of this ctor to be very low so it is one of the last to
    // run.
    appendToGlobalCtors(m, ctor, 65536);
  }
};

} // namespace

void llvm::detail::genCtorCuda(Module &m, detail::GetTLI getTLI,
                               const TapirTargetOptions &tto,
                               const detail::GenerateCtorOptions &genCtorOpts) {
  GenerateCtorCuda(getTLI, tto, genCtorOpts).run(m);
}
