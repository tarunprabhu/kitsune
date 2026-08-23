//===- EmbLowerKitIntrinsics.cpp - Lower Kitsune-specific intrinsics ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune-specific intrinsics in embedded modules.
//
//===----------------------------------------------------------------------===//

#include "kitsune/CodeGen/EmbLowerKitIntrinsics.h"
#include "kitsune/CodeGen/EmbModuleLegacyPass.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/TargetUtils.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/TargetParser.h"

#define DEBUG_TYPE "emb-lower-intrinsics"

using namespace llvm;

namespace {

class LowerKitIntrinsicsBase {
protected:
  const TTOptions &tto;

private:
  bool shouldReplace(Intrinsic::ID intr);
  bool replaceSimple(IRBuilder<> &builder, CallBase *call, Intrinsic::ID intr);

protected:
  virtual bool replaceWarpSize(CallBase *call) = 0;

  // Certain intrinsics should have been lowered already. If they make it here,
  // we should error out now because it will probably result in a failure later
  // anyway, and we can provide a more graceful error message here.
  virtual std::optional<StringRef> getFailMsg(Intrinsic::ID intr) = 0;

  // Some intrinsics have a one-to-one lowering to another target-specific
  // intrinsic.
  virtual std::optional<Intrinsic::ID> getReplIntr(Intrinsic::ID) = 0;

protected:
  LowerKitIntrinsicsBase(const TTOptions &tto) : tto(tto) {}

  const TTOptions &getTTO() const { return tto; }

public:
  bool run(Module &m);
};

class LowerKitIntrinsicsCuda : public LowerKitIntrinsicsBase {
protected:
  virtual bool replaceWarpSize(CallBase *call) override;
  virtual std::optional<StringRef> getFailMsg(Intrinsic::ID) override;
  virtual std::optional<Intrinsic::ID> getReplIntr(Intrinsic::ID) override;

public:
  LowerKitIntrinsicsCuda(const TTOptions &tto) : LowerKitIntrinsicsBase(tto) {}
};

class LowerKitIntrinsicsHip : public LowerKitIntrinsicsBase {
protected:
  virtual bool replaceWarpSize(CallBase *call) override;
  virtual std::optional<StringRef> getFailMsg(Intrinsic::ID) override;
  virtual std::optional<Intrinsic::ID> getReplIntr(Intrinsic::ID) override;

public:
  LowerKitIntrinsicsHip(const TTOptions &tto) : LowerKitIntrinsicsBase(tto) {}
};

} // namespace

//--------------------------- LowerKitIntrinsicsBase ---------------------------

bool LowerKitIntrinsicsBase::shouldReplace(Intrinsic::ID intr) {
  if (intr == Intrinsic::kit_gpu_warp_size)
    return true;
  return getReplIntr(intr).has_value();
}

bool LowerKitIntrinsicsBase::replaceSimple(IRBuilder<> &builder, CallBase *call,
                                           Intrinsic::ID newIntr) {
  builder.SetInsertPoint(call);

  // The first argument of the call will be the TTID. This is never needed when
  // lowering a simple call.
  SmallVector<Value *, 4> args;
  for (unsigned i = 1; i < call->arg_size(); ++i)
    args.push_back(call->getArgOperand(i));

  CallInst *newCall = builder.CreateIntrinsic(newIntr, args);
  newCall->takeName(call);
  call->replaceAllUsesWith(newCall);
  call->eraseFromParent();

  return true;
}

bool LowerKitIntrinsicsBase::run(Module &m) {
  // Some intrinsics should have been lowered already, regardless of the tapir
  // target. If these are encountered here, raise an error.
  static const SmallSet<Intrinsic::ID, 2> unexpected = {};

  SmallVector<CallBase *, 4> calls;
  for (Function &f : m)
    for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
      if (auto *call = dyn_cast<CallBase>(&*i))
        if (Intrinsic::ID intr = call->getIntrinsicID()) {
          if (shouldReplace(intr))
            calls.push_back(call);
          else if (unexpected.contains(intr))
            llvm_unreachable(
                "Intrinsic should have been lowered by an earlier pass");
          else if (std::optional<StringRef> msg = getFailMsg(intr))
            llvm_unreachable(msg->data());
        }

  IRBuilder<> builder(m.getContext());

  bool changed = false;
  for (CallBase *call : calls) {
    Intrinsic::ID intr = call->getIntrinsicID();
    if (std::optional<Intrinsic::ID> replIntr = getReplIntr(intr))
      changed |= replaceSimple(builder, call, *replIntr);
    else if (intr == Intrinsic::kit_gpu_warp_size)
      changed |= replaceWarpSize(call);
    else
      llvm_unreachable("LowerKitIntrinsicsBase::run: Unexpected intrinsic");
  }
  return changed;
}

//--------------------------- LowerKitIntrinsicsCuda ---------------------------

bool LowerKitIntrinsicsCuda::replaceWarpSize(CallBase *call) {
  // On NVIDIA GPU's, the warp size is always 32.
  LLVMContext &ctx = call->getContext();
  call->replaceAllUsesWith(toConstant(32U, ctx));
  call->eraseFromParent();

  return true;
}

std::optional<StringRef> LowerKitIntrinsicsCuda::getFailMsg(Intrinsic::ID) {
  // There are no intrinsics whose presence should trigger a failure.
  return std::nullopt;
}

std::optional<Intrinsic::ID>
LowerKitIntrinsicsCuda::getReplIntr(Intrinsic::ID intr) {
  switch (intr) {
  case Intrinsic::kit_gpu_thread_id_x:
    return Intrinsic::nvvm_read_ptx_sreg_tid_x;
  case Intrinsic::kit_gpu_thread_id_y:
    return Intrinsic::nvvm_read_ptx_sreg_tid_y;
  case Intrinsic::kit_gpu_thread_id_z:
    return Intrinsic::nvvm_read_ptx_sreg_tid_z;
  case Intrinsic::kit_gpu_block_id_x:
    return Intrinsic::nvvm_read_ptx_sreg_ctaid_x;
  case Intrinsic::kit_gpu_block_id_y:
    return Intrinsic::nvvm_read_ptx_sreg_ctaid_y;
  case Intrinsic::kit_gpu_block_id_z:
    return Intrinsic::nvvm_read_ptx_sreg_ctaid_z;
  case Intrinsic::kit_gpu_block_size_x:
    return Intrinsic::nvvm_read_ptx_sreg_ntid_x;
  case Intrinsic::kit_gpu_block_size_y:
    return Intrinsic::nvvm_read_ptx_sreg_ntid_y;
  case Intrinsic::kit_gpu_block_size_z:
    return Intrinsic::nvvm_read_ptx_sreg_ntid_z;
  case Intrinsic::kit_gpu_grid_size_x:
    return Intrinsic::nvvm_read_ptx_sreg_nctaid_x;
  case Intrinsic::kit_gpu_grid_size_y:
    return Intrinsic::nvvm_read_ptx_sreg_nctaid_y;
  case Intrinsic::kit_gpu_grid_size_z:
    return Intrinsic::nvvm_read_ptx_sreg_nctaid_z;
  default: break;
  }
  return std::nullopt;
}

//--------------------------- LowerKitIntrinsicsHip ---------------------------

bool LowerKitIntrinsicsHip::replaceWarpSize(CallBase *call) {
  auto isWave64 = [](AMDGPU::GPUKind kind) -> bool {
    const unsigned archAttrs = AMDGPU::getArchAttrAMDGCN(kind);
    const bool hasWave32 = (archAttrs & AMDGPU::FEATURE_WAVE32);
    return !hasWave32;
  };

  const TargetMachine *tm = createTargetMachine(TTID::Hip, tto);
  assert(tm && "Got an AMDGPU target machine");

  const MCSubtargetInfo *stInfo = tm->getMCSubtargetInfo();
  assert(stInfo && "Got subtarget information");

  // Dealing with the warp size on AMDGPU is tricky. Some devices only support a
  // warp size of 32, others only support 64. But a few support both. Which one
  // is used is determined by either the target features set on the function
  // containing the intrinsic call.
  //
  //   - If the target features have been set, they are always used.
  //
  //   - If the target features are not set, the device architecture is queried
  //     to determine the warp size.
  //
  // If the device architecture is not set correctly, an error is raised. We do
  // not revert to a default because there is no guarantee that the chosen
  // default will work across devices.
  //
  // NOTE: This is unlikely to error out at use-time, but it depends on the
  // frontend setting the correct features. It also depends on a default device
  // being set (getCPU() will return the default hip architecture). If the
  // frontend does not correctly set these features, or if the default
  // architecture being compiled for is incompatible with the GPU being used,
  // the compiled code will not work correctly. It is not clear that we can
  // do a lot about that.
  Function *f = call->getFunction();
  StringRef features = f->getFnAttribute("target-features").getValueAsString();
  unsigned wavefrontSize = 0;
  if (features.contains("+wavefrontsize32"))
    wavefrontSize = 32;
  else if (features.contains("+wavefrontsize64"))
    wavefrontSize = 64;
  else if (AMDGPU::GPUKind kind = AMDGPU::parseArchAMDGCN(stInfo->getCPU()))
    wavefrontSize = isWave64(kind) ? 64 : 32;
  else
    llvm_unreachable("Could not determine wavefront size");

  LLVMContext &ctx = f->getContext();
  call->replaceAllUsesWith(toConstant(wavefrontSize, ctx));
  call->eraseFromParent();
  return true;
}

std::optional<StringRef> LowerKitIntrinsicsHip::getFailMsg(Intrinsic::ID intr) {
  switch (intr) {
  case Intrinsic::kit_gpu_thread_id_x:
  case Intrinsic::kit_gpu_thread_id_y:
  case Intrinsic::kit_gpu_thread_id_z:
  case Intrinsic::kit_gpu_block_id_x:
  case Intrinsic::kit_gpu_block_id_y:
  case Intrinsic::kit_gpu_block_id_z:
  case Intrinsic::kit_gpu_block_size_x:
  case Intrinsic::kit_gpu_block_size_y:
  case Intrinsic::kit_gpu_block_size_z:
  case Intrinsic::kit_gpu_grid_size_x:
  case Intrinsic::kit_gpu_grid_size_y:
  case Intrinsic::kit_gpu_grid_size_z:
    return "GPU thread intrinsics in AMDGPU device modules should have been "
           "replaced by the emb-lower-intrinsics-early pass";
  default: break;
  }
  return std::nullopt;
}

std::optional<Intrinsic::ID>
LowerKitIntrinsicsHip::getReplIntr(Intrinsic::ID intr) {
  // No simple replacements at this time.
  return std::nullopt;
}

//------------------------------------------------------------------------------

static bool lowerKitIntrinsics(Module &embM, TTID tt, const TTOptions &tto) {
  switch (tt) {
  case TTID::Cuda: return LowerKitIntrinsicsCuda(tto).run(embM);
  case TTID::Hip: return LowerKitIntrinsicsHip(tto).run(embM);
  default: break;
  }
  llvm_unreachable("lowerKitIntrinsics: TTID not handled");
}

namespace {

/// Pass, for the legacy pass manager, to lower kitsune-specific intrinsics.
class EmbLowerKitIntrinsicsLegacyPass : public EmbModuleLegacyPass {
public:
  EmbLowerKitIntrinsicsLegacyPass() : EmbModuleLegacyPass(ID) {
    initializeEmbLowerKitIntrinsicsLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Lower Kitsune intrinsics (embedded)";
  }

  void getAnalysisUsage(AnalysisUsage &au) const override {
    au.addRequired<TTObjectsAnalysisWrapperPass>();
  }

  bool runOnEmbModule(TTID tt, Module &embM) override {
    TTObjects ttObjs = getAnalysis<TTObjectsAnalysisWrapperPass>().getResult();
    const TTOptions &tto = ttObjs.getOptions();

    return lowerKitIntrinsics(embM, tt, tto);
  }

public:
  static char ID;
};

} // namespace

char EmbLowerKitIntrinsicsLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(EmbLowerKitIntrinsicsLegacyPass, DEBUG_TYPE,
                      "Lower Kitsune intrinsics (embedded)", false, false)
INITIALIZE_PASS_DEPENDENCY(TTObjectsAnalysisWrapperPass)
INITIALIZE_PASS_END(EmbLowerKitIntrinsicsLegacyPass, DEBUG_TYPE,
                    "Lower Kitsune intrinsics (embedded)", false, false)

ModulePass *llvm::createEmbLowerKitIntrinsicsLegacyPass() {
  return new EmbLowerKitIntrinsicsLegacyPass();
}

bool EmbLowerKitIntrinsicsPass::run(TTID tt, Module &embM, Module &hostM,
                                    ModuleAnalysisManager &hostMAM) {
  const TTObjects &ttObjs = hostMAM.getResult<TTObjectsAnalysis>(hostM);
  const TTOptions &tto = ttObjs.getOptions();

  return lowerKitIntrinsics(embM, tt, tto);
}
