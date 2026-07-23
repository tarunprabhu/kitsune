//===- CGKitAttrs.cpp - Codegen for Kitsune's attributes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// "Codegen" (i.e. LLVM IR generation) for Kitsune's attributes
//
//===----------------------------------------------------------------------===//

#include "CGKitsune.h"
#include "CodeGenFunction.h"
#include "kitsune/Core/KitOptions.h"

using namespace clang;
using namespace CodeGen;

static StringRef getLLVMAttrNameFor(const KitMemAccessAttr &Attr) {
  if (Attr.isWriteOnly())
    return "kit.writeonly";
  else if (Attr.isReadWrite())
    return "kit.readwrite";
  else if (Attr.isReadOnly())
    return "kit.readonly";
  llvm_unreachable("Unknown kitsune memory access attribute");
}

static void addKitMemAccessAttr(CodeGenModule &cgm, const FunctionDecl &fd,
                                llvm::Function &f) {
  llvm::LLVMContext &ctx = f.getContext();

  if (const auto *attr = fd.getAttr<KitMemAccessAttr>())
    f.addFnAttr(getLLVMAttrNameFor(*attr));

  for (unsigned i = 0, n = fd.getNumParams(); i < n; ++i) {
    const ParmVarDecl *param = fd.getParamDecl(i);
    QualType paramTy = param->getType();
    const Decl *pDecl = param;
    if (const auto *typdef = dyn_cast<TypedefType>(paramTy))
      pDecl = typdef->getDecl();

    if (const auto *attr = pDecl->getAttr<KitMemAccessAttr>()) {
      if (paramTy.getTypePtr()->isStructureOrClassType()) {
        cgm.ErrorUnsupported(
            param,
            "cannot handle kitsune memaccess attribute on a struct or class");
        break;
      }

      llvm::Argument *arg = f.getArg(i);
      arg->addAttr(llvm::Attribute::get(ctx, getLLVMAttrNameFor(*attr)));
    }
  }
}

void clang::CodeGen::AddKitAttributes(CodeGenModule &cgm, const VarDecl &vd,
                                      llvm::GlobalVariable &g) {
  // Only set the kitsune-specific attributes if a tapir target has been set.
  // In general, we try to keep the Kitsune-specific additions to clang strictly
  // opt-in features. Since the attributes will have no effect unless lowering
  // using tapir is enabled, we might as well only add the attributes only if a
  // tapir target has been set.
  if (!cgm.getKitOpts().hasTTID())
    return;

  if (const auto *attr = vd.getAttr<KitMemAccessAttr>())
    g.addAttribute(getLLVMAttrNameFor(*attr));
}

void clang::CodeGen::AddKitAttributes(CodeGenModule &cgm,
                                      const FunctionDecl &fd,
                                      llvm::Function &f) {
  // Only set the kitsune-specific attributes if a tapir target has been set.
  // In general, we try to keep the Kitsune-specific additions to clang strictly
  // opt-in features. Since the attributes will have no effect unless lowering
  // using tapir is enabled, we might as well only add the attributes only if a
  // tapir target has been set.
  if (!cgm.getKitOpts().hasTTID())
    return;

  addKitMemAccessAttr(cgm, fd, f);
}
