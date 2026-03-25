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
#include "llvm/IR/Instructions.h"

using namespace llvm;

LLVMContext &llvm::getContext(Instruction &inst) { return inst.getContext(); }

LLVMContext &llvm::getContext(const Instruction &inst) {
  return inst.getContext();
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
