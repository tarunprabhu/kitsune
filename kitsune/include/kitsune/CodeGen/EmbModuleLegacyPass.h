//==- EmbModulePass.h - Embedded module pass for legacy pass mgr -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for passes that operate on embedded modules for the legacy pass
// manager. These typically perform transformations on the embedded modules and
// update the global variables in the parent module that contain them.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CODEGEN_EMB_MODULE_LEGACY_PASS_H
#define KITSUNE_CODEGEN_EMB_MODULE_LEGACY_PASS_H

#include "kitsune/Core/TTID.h"
#include "llvm/Pass.h"

namespace llvm {

class Module;

/// \ingroup kitsune
/// Base class for pass used by the legacy pass manager that operates on
/// embedded modules. This is the equivalent of the \ref EmbModulePass that
/// works with the new pass manager. Embedded module legacy passes must inherit
/// from this class and override the runOnEmbModule method.
///
/// The only change that this class might make to the host module is to change
/// the type and initializer of one or more global variables that contain the
/// embedded modules.
class EmbModuleLegacyPass : public ModulePass {
protected:
  EmbModuleLegacyPass(char ID);

  /// The "entry" method for such passes. The pass manager will invoke the
  /// runOnModule method as usual. This method will, in turn invoke this.
  /// \param tt The tapir target that generated the module.
  /// \param devM The embedded module
  /// \return true if the embedded module was changed, false otherwise.
  virtual bool runOnEmbModule(TTID tt, Module &devM) = 0;

public:
  virtual void getAnalysisUsage(AnalysisUsage &au) const override;

  virtual bool runOnModule(Module &m) override;
};

} // namespace llvm

#endif // KITSUNE_CODEGEN_EMB_MODULE_LEGACY_PASS_H
