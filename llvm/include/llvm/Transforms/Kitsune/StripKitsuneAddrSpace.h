//===- StripKitsuneAddrSpace.h - Strip kitsune address spaces -*- C++ -*----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Currently, Kitsune's frontend uses address spaces as a stand-in for
// attributed pointers which are not easy to implement in LLVM. This pass strips
// those address spaces and puts all the pointers in the default address space.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_KITSUNE_STRIP_ADDR_SPACE_H
#define LLVM_TRANSFORMS_KITSUNE_STRIP_ADDR_SPACE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// Since type attributes are not allowed in LLVM, we denote distinguish between
/// mobile and regular pointers by putting the former in a special address
/// space. For some tapir targets, notably hip, this is a problem because hip
/// expects certain program elements to be in specific address spaces. This
/// pass goes through the entire module and replace all pointers in
/// kitsune-specific address spaces with pointers in the default address space.
class StripKitsuneAddrSpacePass
    : public PassInfoMixin<StripKitsuneAddrSpacePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  /// It is not clear if this pass is actually required.
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_KITSUNE_STRIP_ADDR_SPACE_H
