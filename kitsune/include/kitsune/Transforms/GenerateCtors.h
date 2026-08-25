//===- GenerateCtors.h - Generate global ctors and dtors --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generate Kitsune global ctors and dtors
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_GENERATE_CTORS_H
#define KITSUNE_TRANSFORMS_GENERATE_CTORS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class TTOptions;

/// \ingroup kitsune
/// Generates global constructors and destructors needed by Kitsune.
class GenerateCtorsPass : public PassInfoMixin<GenerateCtorsPass> {
private:
  const TTOptions &tto;

public:
  GenerateCtorsPass(const TTOptions &tto) : tto(tto) {}

  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_GENERATE_CTORS_H
