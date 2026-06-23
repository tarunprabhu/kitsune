//===- EarlyVerification.cpp - Kitsune-specific early verification --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kitsune-specific verification that is carried out early in the optimization
// pipeline.
//
// This is run as early in the optimization pipeline as possible. The checks
// here catch issues that really ought to be caught by the frontends, though,
// admittedly, some of these may require more effort to catch in some frontends.
// This also provides a secondary safety net in case we want to use Kitsune in
// situations where there is no "reasonable" frontend - such as in a JIT
// compilation context.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Analysis/EarlyVerification.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {

class VerifyFunction {
private:
  LoopInfo &li;

private:
  bool verifyReductions(Function &f);

public:
  VerifyFunction(LoopInfo &li) : li(li) {}

  // Return false if at least one error was found in the function, true
  // otherwise.
  bool verify(Function &f);
};

class VerifyModule {
private:
  [[maybe_unused]] ModuleAnalysisManager &mam;
  [[maybe_unused]] FunctionAnalysisManager &fam;

public:
  VerifyModule(ModuleAnalysisManager &mam, FunctionAnalysisManager &fam)
      : mam(mam), fam(fam) {}

  // Return false if at least one error was found in the module, true otherwise.
  bool verify(Module &m);
};

} // namespace

template <typename IR, typename... Args>
static bool complain(const IR &elem, DiagID diagID, Args &&...args) {
  emitDiagnostic(elem, diagID, args...);
  return false;
}

bool VerifyFunction::verifyReductions(Function &f) {
  auto isNestedTapirLoop = [](const Loop &loop) -> bool {
    if (isTapirLoop(loop))
      return isAnyAncestorTapirLoop(loop);
    else if (const Loop *ancestorLoop = getNearestAncestorTapirLoop(loop))
      return isAnyAncestorTapirLoop(*ancestorLoop);
    llvm_unreachable("Loop must have ancestor tapir loop");
  };

  for (Loop *loop : li.getLoopsInPreorder())
    if (isTapirLoop(*loop))
      for (BasicBlock *bb : loop->getBlocks())
        for (Instruction &inst : *bb)
          if (auto *call = dyn_cast<CallBase>(&inst))
            if (call->getIntrinsicID() == Intrinsic::kit_reduce_0)
              if (isNestedTapirLoop(*li.getLoopFor(bb)))
                return complain(*call, DiagID::ErrNYI,
                                "nested parallel reductions");
  return true;
}

bool VerifyFunction::verify(Function &f) {
  bool ok = true;
  ok &= verifyReductions(f);

  return ok;
}

bool VerifyModule::verify(Module &m) {
  bool ok = true;

  // At this time, the only early verifications to be performed are on
  // functions. If we ever need to verify global variables or other module-level
  // entities, do those here.

  for (Function &f : m) {
    if (f.size()) {
      LoopInfo &li = fam.getResult<LoopAnalysis>(f);

      ok &= VerifyFunction(li).verify(f);
    }
  }
  return ok;
}

PreservedAnalyses EarlyVerificationPass::run(Module &m,
                                             ModuleAnalysisManager &mam) {
  FunctionAnalysisManager &fam =
      mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();

  bool ok = VerifyModule(mam, fam).verify(m);
  if (!ok && exitIfError)
    exitOnError();

  return PreservedAnalyses::all();
}
