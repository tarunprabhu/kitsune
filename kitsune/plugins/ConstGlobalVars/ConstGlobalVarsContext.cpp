/*
 * Copyright (c) 2022 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */

#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"

#include "ConstGlobalVarsContext.h"

using namespace clang;

// A singleton instance of the shared context object. It would be nice to be
// able to get rid of this and have it be owned by some other object, but there
// is currently no context that is otherwise shared between the LLVM-IR passes
// and the AST plugin. This is only accessible through the static method on
// ConstGlobalVarsContext, so that's the small saving grace here.
//
static thread_local std::unique_ptr<ConstGlobalVarsContext> singletonContext;

ConstGlobalVarsContext& ConstGlobalVarsContext::getOrCreate() {
  if (not singletonContext)
    singletonContext.reset(new ConstGlobalVarsContext());
  return *singletonContext;
}

const ConstGlobalVarsContext* ConstGlobalVarsContext::getIfExists() {
  return singletonContext.get();
}

void ConstGlobalVarsContext::addConstGlobal(const Decl* D) {
  constGlobalVars.insert(D);
}

bool ConstGlobalVarsContext::isConstGlobal(llvm::StringRef Name) const {
  // The CodeGenerator will be null if no source files were parsed during the
  // compiler invocation (this can happen when linking or if the input files
  // were only bitcode files, AST deserializations etc). However, the
  // corresponding LLVM pass will always be called and that pass will always
  // attempt to check if a global variable is constant.
  if (CG) {
    const Decl* D = CG->GetDeclForMangledName(Name);
    return constGlobalVars.find(D) != constGlobalVars.end();
  }
  return false;
}

CodeGenerator* ConstGlobalVarsContext::makeCodeGenerator(CompilerInstance& CI) {
  CG = CreateLLVMCodeGen(CI.getDiagnostics(),
                         "kitsune-parallel-module",
                         CI.getHeaderSearchOpts(),
                         CI.getPreprocessorOpts(),
                         CI.getCodeGenOpts(),
                         llvmCtxt);
  return CG;
}
