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
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "emb-lower-intrinsics"

using namespace llvm;

namespace {

class LowerKitIntrinsicsBase {
protected:
private:
  bool shouldReplace(Intrinsic::ID intr);
  bool replaceSimple(IRBuilder<> &builder, CallBase *call, Intrinsic::ID intr);

protected:
  // Certain intrinsics should have been lowered already. If they make it here,
  // we should error out now because it will probably result in a failure later
  // anyway, and we can provide a more graceful error message here.
  virtual std::optional<StringRef> getFailMsg(Intrinsic::ID intr) = 0;

  // Some intrinsics have a one-to-one lowering to another target-specific
  // intrinsic.
  virtual std::optional<Intrinsic::ID> getReplIntr(Intrinsic::ID) = 0;

public:
  bool run(Module &m);
};

class LowerKitIntrinsicsCuda : public LowerKitIntrinsicsBase {
protected:
  virtual std::optional<StringRef> getFailMsg(Intrinsic::ID) override;
  virtual std::optional<Intrinsic::ID> getReplIntr(Intrinsic::ID) override;
};

class LowerKitIntrinsicsHip : public LowerKitIntrinsicsBase {
protected:
  virtual std::optional<StringRef> getFailMsg(Intrinsic::ID) override;
  virtual std::optional<Intrinsic::ID> getReplIntr(Intrinsic::ID) override;
};

} // namespace

//--------------------------- LowerKitIntrinsicsBase ---------------------------

bool LowerKitIntrinsicsBase::shouldReplace(Intrinsic::ID intr) {
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
  static const SmallSet<Intrinsic::ID, 2> unexpected = {
      Intrinsic::kit_gpu_warp_id,
      Intrinsic::kit_gpu_warp_lane,
      Intrinsic::kit_gpu_warp_size,
  };

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
    else
      llvm_unreachable("LowerKitIntrinsicsBase::run: Unexpected intrinsic");
  }
  return changed;
}

//--------------------------- LowerKitIntrinsicsCuda ---------------------------

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

static bool lowerKitIntrinsics(Module &embM, TTID tt) {
  switch (tt) {
  case TTID::Cuda: return LowerKitIntrinsicsCuda().run(embM);
  case TTID::Hip: return LowerKitIntrinsicsHip().run(embM);
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

  bool runOnEmbModule(TTID tt, Module &embM) override {
    return lowerKitIntrinsics(embM, tt);
  }

public:
  static char ID;
};

} // namespace

char EmbLowerKitIntrinsicsLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(EmbLowerKitIntrinsicsLegacyPass, DEBUG_TYPE,
                      "Lower Kitsune intrinsics (embedded)", false, false)
INITIALIZE_PASS_END(EmbLowerKitIntrinsicsLegacyPass, DEBUG_TYPE,
                    "Lower Kitsune intrinsics (embedded)", false, false)

ModulePass *llvm::createEmbLowerKitIntrinsicsLegacyPass() {
  return new EmbLowerKitIntrinsicsLegacyPass();
}

bool EmbLowerKitIntrinsicsPass::run(TTID tt, Module &embM, Module &hostM,
                                    ModuleAnalysisManager &hostMAM) {
  return lowerKitIntrinsics(embM, tt);
}
