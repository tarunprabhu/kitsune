//===- StripKitsuneAddrSpaces.h - Strip kitsune address spaces --*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Move pointers in Kitsune-specific address spaces to the default address
// space.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CODEGEN_STRIP_KITSUNE_ADDR_SPACES_H
#define KITSUNE_CODEGEN_STRIP_KITSUNE_ADDR_SPACES_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class ModulePass;

/// \ingroup kitsune
/// Move pointers from Kitsune-specific address spaces to the default address
/// space. This will mutate the types of the appropriate entities.
class StripKitsuneAddrSpacesPass
    : public PassInfoMixin<StripKitsuneAddrSpacesPass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);

  static bool isRequired() { return true; }
};

/// \ingroup kitsune
ModulePass *createStripKitsuneAddrSpacesLegacyPass();

} // namespace llvm

#endif // KITSUNE_CODEGEN_STRIP_KITSUNE_ADDR_SPACES_H
