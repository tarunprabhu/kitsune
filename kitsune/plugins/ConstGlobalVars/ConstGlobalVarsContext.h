/*
 * Copyright (c) 2022 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */

#ifndef KITSUNE_PLUGINS_CONST_GLOBAL_VARS_CONTEXT_H
#define KITSUNE_PLUGINS_CONST_GLOBAL_VARS_CONTEXT_H

#include "llvm/IR/LLVMContext.h"

#include <set>

// Forward declarations for the clang objects because some of the clang headers
// are large enough to noticeably slow down compilation. Therefore, only
// include those that are absolutely necessary.
namespace clang {
  class CodeGenerator;
  class CompilerInstance;
  class Decl;
}

// This class maintains the state that needs to be shared between the frontend
// plugin that operates on the AST and the LLVM IR passes that use the
// additional data from the frontend.
class ConstGlobalVarsContext {
private:
  // The pointers here *may* be freed before the LLVM pass has a chance to
  // run. I don't actually think that is happening, but I haven't been able to
  // figure it out to my satisfaction. Therefore,
  //
  // -------------------------------------------------------------------------
  //    Decl's in this set MUST **NOT** be dereferenced in the LLVM passes
  // -------------------------------------------------------------------------
  //
  std::set<const clang::Decl*> constGlobalVars;

  // This context is only used to create the parallel LLVM module.
  llvm::LLVMContext llvmCtxt;

  // The CodeGenerator will be owned by ... someone else. The CreateASTConsumer
  // method will pass a unique_ptr to this object to a MultiplexConsumer.
  // Since the plugin action is run before the main action, the CodeGenerator
  // instance lives until after LLVM-IR generation which is what we need.
  //
  // NOTE: If no source files are parsed (as may happen during linking), this
  // will be unset.
  clang::CodeGenerator* CG = nullptr;

public:
  clang::CodeGenerator* makeCodeGenerator(clang::CompilerInstance& CI);
  void addConstGlobal(const clang::Decl* D);
  bool isConstGlobal(llvm::StringRef Name) const;

public:
  // Return the singleton instance of this class. This will create the instance
  // if it has not already been created.
  static ConstGlobalVarsContext& getOrCreate();

  // Return a singleton instance of this class if it has already been created.
  // If not, return null. This should only be called from any LLVM passes, so
  // return a const pointer.
  static const ConstGlobalVarsContext* getIfExists();
};

#endif // KITSUNE_PLUGINS_CONST_GLOBAL_VARS_CONTEXT_H
