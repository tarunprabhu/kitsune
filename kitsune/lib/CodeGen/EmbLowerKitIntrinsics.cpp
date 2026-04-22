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
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "emb-lower-intrinsics"

using namespace llvm;

// This is a very simple implementation that assumes that the mapped intrinsic
// has exactly the same signature as the kitsune-specific intrinsic being
// replaced.
static bool
replaceAllSimple(Module &embM,
                 std::function<Intrinsic::ID(Intrinsic::ID)> getNewCallee) {
  SmallDenseMap<CallBase *, Intrinsic::ID, 16> calls;
  for (Function &f : embM)
    for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
      if (auto *call = dyn_cast<CallBase>(&*i))
        if (Intrinsic::ID callee = call->getIntrinsicID())
          if (Intrinsic::ID newCallee = getNewCallee(callee))
            calls.insert({call, newCallee});

  for (auto [call, newCallee] : calls) {
    IRBuilder<> builder(call);
    SmallVector<Value *, 4> args(call->arg_begin(), call->arg_end());
    CallInst *newCall = builder.CreateIntrinsic(newCallee, args);

    newCall->takeName(call);
    call->replaceAllUsesWith(newCall);
    call->eraseFromParent();
  }

  return calls.size();
}

static bool lowerKitsuneCudaIntrinsics(Module &embM) {
  auto getNewCallee = [](Intrinsic::ID id) -> Intrinsic::ID {
    switch (id) {
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
    default:
      return Intrinsic::not_intrinsic;
    }
  };
  return replaceAllSimple(embM, getNewCallee);
}

static bool lowerKitsuneHipIntrinsics(Module &embM) {
  auto getNewCallee = [](Intrinsic::ID id) -> Intrinsic::ID {
    switch (id) {
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
      llvm_unreachable(
          "GPU thread intrinsics in AMDGPU device modules should have been "
          "replaced by the emb-lower-intrinsics-early pass");
    default:
      return Intrinsic::not_intrinsic;
    }
  };
  return replaceAllSimple(embM, getNewCallee);
}

static bool lowerKitIntrinsics(TTID tt, Module &embM) {
  switch (tt) {
  case TTID::Cuda:
    return lowerKitsuneCudaIntrinsics(embM);
  case TTID::Hip:
    return lowerKitsuneHipIntrinsics(embM);
  default:
    break;
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
    return lowerKitIntrinsics(tt, embM);
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
                                    ModuleAnalysisManager &hosMAM) {
  return lowerKitIntrinsics(tt, embM);
}
