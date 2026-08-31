//=- StripKitAddrSpaces.h - Strip Kitsune-specific address spaces -*- C++ -*-=//
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

#ifndef KITSUNE_CODEGEN_STRIP_KIT_ADDR_SPACES_H
#define KITSUNE_CODEGEN_STRIP_KIT_ADDR_SPACES_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class ModulePass;
class PassRegistry;

/// \ingroup kitsune
/// @{

/// Move pointers from Kitsune-specific address spaces to the default address
/// space. This will mutate the types of the appropriate entities.
class StripKitAddrSpacesPass : public PassInfoMixin<StripKitAddrSpacesPass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);
};

/// Create a legacy pass to strip Kitsune-specific address spaces from a host
/// module.
ModulePass *createStripKitAddrSpacesLegacyPass();
void initializeStripKitAddrSpacesLegacyPassPass(PassRegistry &);

/// @}

} // namespace llvm

#endif // KITSUNE_CODEGEN_STRIP_KIT_ADDR_SPACES_H
