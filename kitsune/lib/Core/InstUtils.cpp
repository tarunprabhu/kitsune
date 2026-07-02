//===- InstUtils.cpp - Utilities for LLVM instructions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM instructions.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/ValueUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

LLVMContext &llvm::getContext(const Instruction &inst) {
  return inst.getContext();
}

Module *llvm::getModule(Instruction &inst) {
  if (BasicBlock *bb = inst.getParent())
    if (Function *f = bb->getParent())
      return f->getParent();
  return nullptr;
}

const Module *llvm::getModule(const Instruction &inst) {
  if (const BasicBlock *bb = inst.getParent())
    if (const Function *f = bb->getParent())
      return f->getParent();
  return nullptr;
}

std::string llvm::getName(const Instruction &inst) {
  auto toString = [](const Instruction &inst) -> std::string {
    SmallString<32> buf;
    raw_svector_ostream os(buf);

    os << inst;
    return buf.str().ltrim().str();
  };

  if (inst.hasName())
    return inst.getName().str();

  std::string buf;
  raw_string_ostream os(buf);

  if (inst.getType()->isVoidTy()) {
    if (const auto *call = dyn_cast<CallBase>(&inst))
      os << "<call " << getName(*call->getCalledOperand()) << ">";
    else
      os << "<" << toString(inst) << ">";
  } else {
    inst.printAsOperand(os, /*PrintType=*/false, inst.getModule());
  }

  return buf;
}

StringRef llvm::getInstClassName(const Instruction &inst) {
  switch (inst.getOpcode()) {
#define HANDLE_INST(OPCODE, CLASS)                                             \
  case Instruction::OPCODE:                                                    \
    return #CLASS;
#define HANDLE_TERM_INST(NUM, OPCODE, CLASS) HANDLE_INST(OPCODE, CLASS)
#define HANDLE_UNARY_INST(NUM, OPCODE, CLASS) HANDLE_INST(OPCODE, CLASS)
#define HANDLE_BINARY_INST(NUM, OPCODE, CLASS) HANDLE_INST(OPCODE, CLASS)
#define HANDLE_MEMORY_INST(NUM, OPCODE, CLASS) HANDLE_INST(OPCODE, CLASS)
#define HANDLE_CAST_INST(NUM, OPCODE, CLASS) HANDLE_INST(OPCODE, CLASS)
#define HANDLE_FUNCLETPAD_INST(NUM, OPCODE, CLASS) HANDLE_INST(OPCODE, CLASS)
#define HANDLE_OTHER_INST(NUM, OPCODE, CLASS) HANDLE_INST(OPCODE, CLASS)
#include "llvm/IR/Instruction.def"
  }
  llvm_unreachable("getClassName: Instruction opcode not handled");
}

bool llvm::isCallSyncRegionStart(const Instruction &inst) {
  if (const auto *call = dyn_cast<CallBase>(&inst))
    if (const Function *f = call->getCalledFunction())
      if (f->getIntrinsicID() == Intrinsic::syncregion_start)
        return true;
  return false;
}

static bool replaceOperand(Instruction &inst, unsigned i, Value *v) {
  assert(inst.getOperand(i)->getType() == v->getType() &&
         "Type mismatch between exiting and new operands");

  bool isSame = inst.getOperand(i) == v;
  inst.setOperand(i, v);

  return !isSame;
}

static bool canReplaceOperands(Instruction &inst) {
  return isa<BinaryOperator>(inst) || isa<CmpInst>(inst) ||
         isa<ReturnInst>(inst) || isa<SelectInst>(inst) ||
         isa<UnaryOperator>(inst);
}

bool llvm::replaceNonMatchingOperands(Instruction &inst, Value *match,
                                      Value *v) {
  // For now, we limit this to instructions where we know that this is safe.
  assert(canReplaceOperands(inst) &&
         "replaceOtherOperands not tested with instruction type");

  bool changed = false;
  for (unsigned i = 0; i < inst.getNumOperands(); ++i)
    if (inst.getOperand(i) != match)
      changed |= replaceOperand(inst, i, v);
  return changed;
}

bool llvm::replaceMatchingOperands(Instruction &inst, Value *match, Value *v) {
  // For now, we limit this to instructions where we know that this is safe.
  assert(canReplaceOperands(inst) &&
         "replaceOtherOperands not tested with instruction type");

  bool changed = false;
  for (unsigned i = 0; i < inst.getNumOperands(); ++i)
    if (inst.getOperand(i) == match)
      changed |= replaceOperand(inst, i, v);
  return changed;
}

Value *llvm::getNonMatchingOperand(BinaryOperator &binOp, Value *match) {
  Value *op0 = binOp.getOperand(0);
  Value *op1 = binOp.getOperand(1);
  if (op0 == match && op1 != match)
    return op1;
  else if (op1 == match && op0 != match)
    return op0;
  return nullptr;
}

BasicBlock *llvm::getNonMatchingSuccessor(Instruction &inst, BasicBlock *bb) {
  if (!inst.isTerminator() || inst.getNumSuccessors() != 2)
    return nullptr;
  else if (inst.getSuccessor(0) == bb)
    return inst.getSuccessor(1);
  else
    return inst.getSuccessor(0);
}

bool llvm::isCondBr(const Instruction &inst) {
  if (const auto *br = dyn_cast<BranchInst>(&inst))
    if (br->isConditional())
      return true;
  return false;
}

bool llvm::isUncondBr(const Instruction &inst) {
  if (const auto *br = dyn_cast<BranchInst>(&inst))
    if (!br->isConditional())
      return true;
  return false;
}
