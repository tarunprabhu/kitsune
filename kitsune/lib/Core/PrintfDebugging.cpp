//===- PrintfDebugging.cpp - Utilities for printf-debugging ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to support "printf-debugging" of Kitsune's transformation passes.
// These make it easy to add printf calls to LLVM-IR. They are intended to be
// useful during development of passes - it is unlikely that one will ever find
// uses of these utilities in code that is not being actively developed or
// debugged.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/PrintfDebugging.h"
#include "kitsune/Core/BasicBlockUtils.h"
#include "kitsune/Core/ConstantUtils.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace llvm;

/// Get or insert a declaration for fprintf in the module \p m.
static FunctionCallee getOrInsertPrintfDecl(Module &m) {
  LLVMContext &ctx = m.getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  Type *ptr = PointerType::getUnqual(ctx);
  FunctionType *fty = FunctionType::get(i32, {ptr, ptr}, /*IsVarArg=*/true);

  return m.getOrInsertFunction("fprintf", fty);
}

/// Get or insert a declaration for a libc stream variable. The name of the
/// variable, \p stream must be either "stdout", or "stderr".
static GlobalVariable *getOrInsertIOStreamDecl(StringRef stream, Module &m) {
  if (GlobalVariable *g = m.getGlobalVariable(stream))
    return g;

  LLVMContext &ctx = m.getContext();
  Type *ptr = PointerType::getUnqual(ctx);
  return new GlobalVariable(m, ptr, /*isConstant=*/false,
                            GlobalValue::ExternalLinkage,
                            /*Initializer=*/nullptr, stream);
}

static Value *insertPrintfImpl(InsertPosition insertPt, StringRef stream,
                               StringRef fmt, ArrayRef<Value *> args,
                               StringRef name) {
  assert(insertPt.getBasicBlock() &&
         "insertBefore must be set to a valid basic block");

  Module *m = getModule(*insertPt.getBasicBlock());
  assert(m && "insertBefore must be set to a basic block in a module");

  LLVMContext &ctx = m->getContext();
  PointerType *ptr = PointerType::getUnqual(ctx);

  FunctionCallee fprintf = getOrInsertPrintfDecl(*m);
  GlobalVariable *strmDecl = getOrInsertIOStreamDecl(stream, *m);
  GlobalVariable *fprintfFmt = createConstString(fmt, *m);

  LoadInst *strm = new LoadInst(ptr, strmDecl, "", insertPt);
  SmallVector<Value *, 4> fprintfArgs = {strm, fprintfFmt};
  for (Value *arg : args)
    fprintfArgs.push_back(arg);

  return CallInst::Create(fprintf, fprintfArgs, name, insertPt);
}

static Value *insertPrintfImpl(IRBuilder<> &builder, StringRef stream,
                               StringRef fmt, ArrayRef<Value *> args,
                               StringRef name) {
  assert(builder.GetInsertBlock() && "Builder must have a valid insert point");

  return insertPrintfImpl(builder.GetInsertPoint(), stream, fmt, args, name);
}

Value *llvm::insertPrintStdout(IRBuilder<> &builder, StringRef fmt,
                               ArrayRef<Value *> args, StringRef name) {
  return insertPrintfImpl(builder, "stdout", fmt, args, name);
}

Value *llvm::insertPrintStderr(IRBuilder<> &builder, StringRef fmt,
                               ArrayRef<Value *> args, StringRef name) {
  return insertPrintfImpl(builder, "stderr", fmt, args, name);
}

Value *llvm::insertPrintStdout(InsertPosition insertPt, StringRef fmt,
                               ArrayRef<Value *> args, StringRef name) {
  return insertPrintfImpl(insertPt, "stdout", fmt, args, name);
}

Value *llvm::insertPrintStderr(InsertPosition insertPt, StringRef fmt,
                               ArrayRef<Value *> args, StringRef name) {
  return insertPrintfImpl(insertPt, "stderr", fmt, args, name);
}
