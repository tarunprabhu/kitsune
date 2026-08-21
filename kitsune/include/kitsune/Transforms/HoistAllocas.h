//==- HoistAllocas.h - Hoist allocas to the function entry block -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hoist allocas to the entry block of the function.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_HOIST_ALLOCAS_H
#define KITSUNE_TRANSFORMS_HOIST_ALLOCAS_H

#include "kitsune/Passes/EmbModulePass.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Hoist allocas to the entry block of the function.
class HoistAllocasPass : public PassInfoMixin<HoistAllocasPass> {
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &am);
};

/// \ingroup kitsune
/// Hoist allocas to the entry block of functions in an embedded bitcode
/// module.
class EmbHoistAllocasPass : public EmbModulePass<EmbHoistAllocasPass> {
public:
  bool run(TTID tt, Module &devM, Module &hostM, ModuleAnalysisManager &hostAM);

  using EmbModulePass<EmbHoistAllocasPass>::run;
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_HOIST_ALLOCAS_H
