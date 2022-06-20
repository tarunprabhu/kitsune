/*
 * Copyright (c) 2022 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */

#ifndef KITSUNE_PLUGINS_CONST_GLOBAL_VARS_PLUGIN_H
#define KITSUNE_PLUGINS_CONST_GLOBAL_VARS_PLUGIN_H

#include "clang/Frontend/FrontendAction.h"

namespace clang {
  class ASTConsumer;
  class CompilerInstance;
} // namespace clang

// This plugin identifies those global variables that are const-qualified
// in the source. It is coupled with an LLVM pass that adds metadata to the
// corresponding globals in the IR indicating that they have been declared
// to be const in the source. This is needed for a couple of reasons:
//
//   1 - When an extern const variable is used in the source, the
//       corresponding global variable in the IR will not be an constant
//       (i.e. GlobalVariable::isConstant() will return false). This is because
//       LLVM requires a constant GlobalVariable to have an initializer whose
//       value is known at compile-time. An extern variable's initializer may
//       be in a different compilation unit and is, therefore, not available
//       at compile-time.
//
//   2 - When dealing with code such as the following file-level variable in C:
//
//           static const var = rand();
//
//       LLVM will not treat var as a constant. Since rand() cannot be
//       evaluated at compile-time, the initialization of var is deferred to
//       a ctor i.e. a global static constructor. LLVM, therefore, sees that
//       the global variable has non-zero writes - the one write in the ctor -
//       and is therefore not a constant, by definition.
//
// How the metadata gets used is unspecified.
//
class ConstGlobalVarsPlugin : public clang::PluginASTAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance& CI, clang::StringRef File) override;

  virtual bool ParseArgs(const clang::CompilerInstance& CI,
                         const std::vector<std::string>& Args) override;

  virtual ActionType getActionType() override;
};

#endif // KITSUNE_PLUGINS_CONST_GLOBAL_VARS_PLUGIN_H
