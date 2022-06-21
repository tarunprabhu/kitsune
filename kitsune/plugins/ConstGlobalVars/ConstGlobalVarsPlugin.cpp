/*
 * Copyright (c) 2022 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/MultiplexConsumer.h"

#include "ConstGlobalVarsPlugin.h"
#include "ConstGlobalVarsContext.h"

using namespace clang;

class Visitor : public clang::RecursiveASTVisitor<Visitor> {
private:
  ConstGlobalVarsContext& SharedCtxt;

private:
  bool isGlobalVar(clang::Decl* D) {
    if (VarDecl* Var = dyn_cast<VarDecl>(D))
      return Var->hasGlobalStorage();
    return false;
  }

  bool isConstGlobalVar(clang::Decl* D) {
    return isGlobalVar(D) and cast<VarDecl>(D)->getType().isConstQualified();
  }

public:
  explicit Visitor(ConstGlobalVarsContext& SharedCtxt)
    : SharedCtxt(SharedCtxt) {
    ;
  }

  // These are declarations of variables.
  bool VisitVarDecl(clang::VarDecl* Var) {
    if (isConstGlobalVar(Var))
      SharedCtxt.addConstGlobal(Var);
    return true;
  }

  // For some reason, when compiling C++, the non-class type top-level
  // declarations are not visited. In that case, we need to to find all uses
  // of global variables and save them if they are constants. When Decl's are
  // used in Expr's, they are wrapped in a DeclRefExpr object.
  bool VisitDeclRefExpr(clang::DeclRefExpr* E) {
    Decl* D = E->getDecl();
    if (isConstGlobalVar(D))
      SharedCtxt.addConstGlobal(D);
    return true;
  }
};

class Consumer : public clang::ASTConsumer {
private:
  ConstGlobalVarsContext &SharedCtxt;

public:
  explicit Consumer(ConstGlobalVarsContext &SharedCtxt)
    : SharedCtxt(SharedCtxt) {
    ;
  }

  virtual ~Consumer() = default;

  virtual void HandleTranslationUnit(clang::ASTContext& Ctxt) override {
    // Even if there are parse errors, this function will still be called. There
    // is no point traversing the AST in that case.
    if (Ctxt.getDiagnostics().getNumErrors() == 0)
      if (TranslationUnitDecl* TU = Ctxt.getTranslationUnitDecl())
        Visitor (SharedCtxt).TraverseDecl(TU);
  }
};

std::unique_ptr<ASTConsumer>
ConstGlobalVarsPlugin::CreateASTConsumer(CompilerInstance& CI,
                                         StringRef) {
  std::vector<std::unique_ptr<ASTConsumer>> Consumers;
  ConstGlobalVarsContext& SharedCtxt = ConstGlobalVarsContext::getOrCreate();

  // In order to be able to find the AST Decl for a llvm::GlobalVariable,
  // we need to match up the (possibly) mangled name of the global with the
  // Decl. The CodeGenerator object has a GetDeclForMangledName() method that
  // will do this. However, the underlying CodeGenerator object is not
  // exposed.
  //
  // It ought to be possible to use a MangleContext object but the default
  // implementation of ItaniumABI has some some hideous corner with respect
  // to constructors, lambdas and templates making it tricky to use reliably.
  //
  // The only option then seems to be to construct a second CodeGenerator
  // which will generate a second LLVM Module. This is patently wasteful
  // since that module will just be thrown away, but I cannot think of another
  // way around this.
  //
  // An alternative may be to replace the default action, but I am not sure
  // what that would entail since there are more than a dozen different
  // supported actions.
  //
  Consumers.emplace_back(SharedCtxt.makeCodeGenerator(CI));
  Consumers.emplace_back(new Consumer(SharedCtxt));

  // By creating a MultiplexConsumer, both the CodeGenerator and the Consumer
  // objects are traversed simultaneously.
  //
  return std::make_unique<MultiplexConsumer>(std::move(Consumers));
}

// This plugin does not support any arguments. If any are supplied, raise
// an error. Returning false from this function will do that.
bool ConstGlobalVarsPlugin::ParseArgs(const CompilerInstance&,
                                      const std::vector<std::string>& Args) {
  return Args.empty();
}

// Overriding this function ensures that the we control when the action is
// run. The default is for the action to run after the main action (which is
// usually CodeGen i.e. LLVM-IR generation). But since we want to modify
// the IR that is generated, we need this to run before the main action.
PluginASTAction::ActionType ConstGlobalVarsPlugin::getActionType() {
  return PluginASTAction::ActionType::AddBeforeMainAction;
}
