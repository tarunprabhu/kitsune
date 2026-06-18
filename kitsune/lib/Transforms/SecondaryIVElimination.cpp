//===- SecondaryIVElimination.cpp - Eliminate secondary indvars -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Eliminate non-primary induction variables from tapir loops. This ensures
// that all tapir loops have exactly one induction variable. This can only
// replace secondary induction variables that are non-side-effecting functions
// of the canonical induction variable of the tapir loop.
//
// NOTE: The current implementation is more restrictive than it needs to be. It
// only supports replacing secondary induction variables, siv, whose updated
// value, siv', is of the form
//
//     siv' = siv OP c
//
// where OP is a supported binary operator and c is a compile-time constant.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/SecondaryIVElimination.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/UseUtils.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#define DEBUG_TYPE "kit-ive"

using namespace llvm;

namespace {

// Main implementation class to eliminate secondary induction variables.
class SecondaryIVElimination {
private:
  LoopInfo &li;

private:
  bool check(Loop &loop);

  bool eliminate(PHINode &iv, Value *repl, Loop &loop);

  bool tryToEliminateFPAdd(PHINode &secIV, Constant *step, Loop &loop);
  bool tryToEliminateFPSub(PHINode &secIV, Constant *step, Loop &loop);
  bool tryToEliminateFPMul(PHINode &secIV, Constant *step, Loop &loop);
  bool tryToEliminateFPDiv(PHINode &secIV, Constant *step, Loop &loop);
  bool tryToEliminateFPRem(PHINode &secIV, Constant *step, Loop &loop);

  bool tryToEliminateIntAdd(PHINode &secIV, Constant *step, Loop &loop);
  bool tryToEliminateIntSub(PHINode &secIV, Constant *step, Loop &loop);
  bool tryToEliminateIntMul(PHINode &secIV, Constant *step, Loop &loop);
  bool tryToEliminateIntDiv(PHINode &secIV, Constant *step, bool isSigned,
                            Loop &loop);
  bool tryToEliminateIntRem(PHINode &secIV, Constant *step, bool isSigned,
                            Loop &loop);
  bool tryToEliminateIntLShl(PHINode &secIV, Constant *step, Loop &loop);
  bool tryToEliminateIntLShr(PHINode &secIV, Constant *step, Loop &loop);
  bool tryToEliminateIntAShr(PHINode &secIV, Constant *step, Loop &loop);

  bool tryToEliminate(PHINode &secIV, Loop &loop);

public:
  SecondaryIVElimination(LoopInfo &li) : li(li) {}

  bool run(Loop &loop);
};

} // namespace

template <typename T, typename... Args>
[[noreturn]] static bool complain(T &elem, DiagID diag, Args &&...args) {
  emitDiagnostic(elem, diag, args...);
  exitOnError();
}

static IRBuilder<> getIRBuilder(Loop &loop) {
  BasicBlock *body = getTapirLoopDetachedBlock(loop);
  return IRBuilder<>(&*body->getFirstNonPHIOrDbg());
}

bool SecondaryIVElimination::eliminate(PHINode &secIV, Value *repl,
                                       Loop &loop) {
  LLVM_DEBUG(dbgs() << "SecIVE:   Eliminating secondary induction '"
                    << getName(secIV) << "'\n");

  BasicBlock *latch = loop.getLoopLatch();
  secIV.replaceUsesWithIf(repl,
                          [latch](Use &u) { return !isUseInBlock(u, *latch); });

  // The only remaining use of the secondary IV will be the increment in the
  // loop latch.
  Value *secIVNext = secIV.getIncomingValueForBlock(latch);

  // There is a cyclic dependence between the secIV definition and its uses
  // since the use is an incoming value in the definition. To remove the use,
  // we must break the cycle. We do this by replacing the incoming value with
  // a constant.
  secIV.setIncomingValueForBlock(latch, getZero(secIV.getType()));

  // Now, we can finally erase everything.
  cast<Instruction>(secIVNext)->eraseFromParent();
  secIV.eraseFromParent();

  return true;
}

// Replace a secondary induction variable, `siv` with initial value `init`,
// step `step`, and type `type`. Here `type` is a floating-point type, and the
// updated value `siv'` is:
//
//     siv' = siv + step
//
// `siv` can be computed from `civ`, the canonical induction variable of the
// loop as shown:
//
//     siv = init + (type)civ * step
//
bool SecondaryIVElimination::tryToEliminateFPAdd(PHINode &secIV, Constant *step,
                                                 Loop &loop) {
  LLVM_DEBUG(dbgs() << "SecIVE:   Floating point addition induction with step '"
                    << *step << "'\n");

  BasicBlock *ph = loop.getLoopPreheader();
  Value *secInit = secIV.getIncomingValueForBlock(ph);
  Type *secType = secIV.getType();
  PHINode *primIV = loop.getCanonicalInductionVariable();

  IRBuilder<> builder = getIRBuilder(loop);
  Value *cstPIV = builder.CreateSIToFP(primIV, secType);
  Value *stride = builder.CreateFMul(cstPIV, step);
  Value *repl = builder.CreateFAdd(secInit, stride);

  return eliminate(secIV, repl, loop);
}

// Replace a secondary induction variable, `siv` with initial value `init`,
// step `step`, and type `type`. Here `type` is a floating-point type, and the
// updated value `siv'` is:
//
//     siv' = siv - step
//
// `siv` can be computed from `civ`, the canonical induction variable of the
// loop as shown:
//
//     siv = init - (type)civ * step
//
bool SecondaryIVElimination::tryToEliminateFPSub(PHINode &secIV, Constant *step,
                                                 Loop &loop) {
  LLVM_DEBUG(
      dbgs() << "SecIVE:   Floating point subtraction induction with step '"
             << *step << "'\n");

  BasicBlock *ph = loop.getLoopPreheader();
  Value *secInit = secIV.getIncomingValueForBlock(ph);
  Type *secType = secIV.getType();
  PHINode *primIV = loop.getCanonicalInductionVariable();

  IRBuilder<> builder = getIRBuilder(loop);
  Value *cstPIV = builder.CreateSIToFP(primIV, secType);
  Value *stride = builder.CreateFMul(cstPIV, step);
  Value *repl = builder.CreateFSub(secInit, stride);

  return eliminate(secIV, repl, loop);
}

// Replace a secondary induction variable, `siv` with initial value `init`,
// step `step`, and type `type`. Here `type` is a floating-point type, and the
// updated value `siv'` is:
//
//     siv' = siv * step
//
// `siv` can be computed from `civ`, the canonical induction variable of the
// loop as shown:
//
//     siv = init * step^civ
//
// where `^` is the exponentiation operator.
//
bool SecondaryIVElimination::tryToEliminateFPMul(PHINode &secIV, Constant *step,
                                                 Loop &loop) {
  LLVM_DEBUG(
      dbgs() << "SecIVE:   Floating point multiplication induction with step '"
             << *step << "'\n");

  BasicBlock *ph = loop.getLoopPreheader();
  Value *secInit = secIV.getIncomingValueForBlock(ph);
  Type *secType = secIV.getType();
  PHINode *primIV = loop.getCanonicalInductionVariable();
  Type *primType = primIV->getType();

  IRBuilder<> builder = getIRBuilder(loop);
  Value *stride = builder.CreateIntrinsic(Intrinsic::powi, {secType, primType},
                                          {step, primIV});
  Value *repl = builder.CreateFMul(secInit, stride);

  return eliminate(secIV, repl, loop);
}

// Replace a secondary induction variable, `siv` with initial value `init`,
// step `step`, and type `type`. Here `type` is a floating-point type, and the
// updated value `siv'` is:
//
//     siv' = siv / step
//
// `siv` can be computed from `civ`, the canonical induction variable of the
// loop as shown:
//
//     siv = init / step^civ
//
// where `^` is the exponentiation operator.
//
bool SecondaryIVElimination::tryToEliminateFPDiv(PHINode &secIV, Constant *step,
                                                 Loop &loop) {
  LLVM_DEBUG(
      dbgs() << "SecIVE:   Floating point remainder induction with step '"
             << *step << "'\n");

  BasicBlock *ph = loop.getLoopPreheader();
  Value *secInit = secIV.getIncomingValueForBlock(ph);
  Type *secType = secIV.getType();
  PHINode *primIV = loop.getCanonicalInductionVariable();
  Type *primType = primIV->getType();

  IRBuilder<> builder = getIRBuilder(loop);
  Value *stride = builder.CreateIntrinsic(Intrinsic::powi, {secType, primType},
                                          {step, primIV});
  Value *repl = builder.CreateFDiv(secInit, stride);

  return eliminate(secIV, repl, loop);
}

// Replace a secondary induction variable, `siv` with initial value `init`,
// step `step`, and type `type`. Here `type` is an integer type, and the updated
// value `siv'` is:
//
//     siv' = siv + step
//
// `siv` can be computed from `civ`, the canonical induction variable of the
// loop as shown:
//
//     siv = init + step*civ
//
bool SecondaryIVElimination::tryToEliminateIntAdd(PHINode &secIV,
                                                  Constant *step, Loop &loop) {
  LLVM_DEBUG(dbgs() << "SecIVE:   Integer addition induction with step '"
                    << *step << "'\n");

  BasicBlock *ph = loop.getLoopPreheader();
  Value *secInit = secIV.getIncomingValueForBlock(ph);
  Type *secType = secIV.getType();
  PHINode *primIV = loop.getCanonicalInductionVariable();

  IRBuilder<> builder = getIRBuilder(loop);
  Value *cstPIV = builder.CreateIntCast(primIV, secType, /*IsSigned=*/false);
  Value *stride = builder.CreateMul(cstPIV, step);
  Value *repl = builder.CreateAdd(secInit, stride);

  return eliminate(secIV, repl, loop);
}

// Replace a secondary induction variable, `siv` with initial value `init`,
// step `step`, and type `type`. Here `type` is an integer type, and the updated
// value `siv'` is:
//
//     siv' = siv - step
//
// `siv` can be computed from `civ`, the canonical induction variable of the
// loop as shown:
//
//     siv = init - step*civ
//
bool SecondaryIVElimination::tryToEliminateIntSub(PHINode &secIV,
                                                  Constant *step, Loop &loop) {
  LLVM_DEBUG(dbgs() << "SecIVE:   Integer subtraction induction with step '"
                    << *step << "'\n");

  BasicBlock *ph = loop.getLoopPreheader();
  Value *secInit = secIV.getIncomingValueForBlock(ph);
  Type *secType = secIV.getType();
  PHINode *primIV = loop.getCanonicalInductionVariable();

  IRBuilder<> builder = getIRBuilder(loop);
  Value *cstPIV = builder.CreateIntCast(primIV, secType, /*IsSigned=*/false);
  Value *stride = builder.CreateMul(cstPIV, step);
  Value *repl = builder.CreateSub(secInit, stride);

  return eliminate(secIV, repl, loop);
}

// Replace a secondary induction variable, `siv` with initial value `init`,
// step `step`, and type `type`. Here `type` is an integer type, and the updated
// value `siv'` is:
//
//     siv' = siv * step
//
// `siv` can be computed from `civ`, the canonical induction variable of the
// loop as shown:
//
//     siv = init * step^civ
//
// where `^` is the exponentiation operator.
//
bool SecondaryIVElimination::tryToEliminateIntMul(PHINode &secIV,
                                                  Constant *step, Loop &loop) {
  LLVM_DEBUG(dbgs() << "SecIVE:   Integer multiplication induction with step '"
                    << *step << "'\n");

  // We can't just use the same approach as floating-point mul here because LLVM
  // does not have an integer power intrinsic, and neither libc, nor libm,
  // provide an integer power function. It is not too difficult to write one of
  // our own - perhaps as part of Kitsune's runtime, but we don't have one at
  // this time.
  //
  // It may be a good idea to implement an integer power intrinsic in Kitsune.
  // The implementation of this transformation would then be similar to the
  // transformation for secondary floating point mul inductions.
  emitDiagnostic(secIV, DiagID::ErrNYI,
                 "eliminate secondary int mul induction from tapir loops");
  exitOnError();
}

// Replace a secondary induction variable, `siv` with initial value `init`,
// step `step`, and type `type`. Here `type` is an integer type, and the updated
// value `siv'` is:
//
//     siv' = siv / step
//
// `siv` can be computed from `civ`, the canonical induction variable of the
// loop as shown:
//
//     siv = init / step^civ
//
// where `^` is the exponentiation operator.
//
bool SecondaryIVElimination::tryToEliminateIntDiv(PHINode &secIV,
                                                  Constant *step, bool isSigned,
                                                  Loop &loop) {
  LLVM_DEBUG(dbgs() << "SecIVE:   Integer division induction with step '"
                    << *step << "'\n");

  // If we implement an integer pow intrinsic in Kitsune, this will become
  // fairly easy to support since we can simply use a transformation similar to
  // the floating point div inductions.
  emitDiagnostic(secIV, DiagID::ErrNYI,
                 "eliminate secondary int div induction from tapir loops");
  exitOnError();
}

// Replace a secondary induction variable, `siv` with initial value `init`,
// step `step`, and type `type`. Here `type` is an integer type, and the updated
// value `siv'` is:
//
//     siv' = siv << step
//
// where `<<` is the logical left shift operator. `siv` can be computed from
// `civ`, the canonical induction variable of the loop as shown:
//
//     siv = init << step*civ
//
bool SecondaryIVElimination::tryToEliminateIntLShl(PHINode &secIV,
                                                   Constant *step, Loop &loop) {
  LLVM_DEBUG(dbgs() << "SecIVE:   Logical shift left induction with step '"
                    << *step << "'\n");

  BasicBlock *ph = loop.getLoopPreheader();
  Value *secInit = secIV.getIncomingValueForBlock(ph);
  Type *secType = secIV.getType();
  PHINode *primIV = loop.getCanonicalInductionVariable();

  IRBuilder<> builder = getIRBuilder(loop);
  Value *cstPIV = builder.CreateIntCast(primIV, secType, /*IsSigned=*/false);
  Value *stride = builder.CreateMul(cstPIV, step);
  Value *repl = builder.CreateShl(secInit, stride);

  return eliminate(secIV, repl, loop);
}

// Replace a secondary induction variable, `siv` with initial value `init`,
// step `step`, and type `type`. Here `type` is an integer type, and the updated
// value `siv'` is:
//
//     siv' = siv >> step
//
// where `>>` is the logical right shift operator. `siv` can be computed from
// `civ`, the canonical induction variable of the loop as shown:
//
//     siv = init >> step*civ
//
bool SecondaryIVElimination::tryToEliminateIntLShr(PHINode &secIV,
                                                   Constant *step, Loop &loop) {
  LLVM_DEBUG(dbgs() << "SecIVE:   Logical shift right induction with step '"
                    << *step << "'\n");

  BasicBlock *ph = loop.getLoopPreheader();
  Value *secInit = secIV.getIncomingValueForBlock(ph);
  Type *secType = secIV.getType();
  PHINode *primIV = loop.getCanonicalInductionVariable();

  IRBuilder<> builder = getIRBuilder(loop);
  Value *cstPIV = builder.CreateIntCast(primIV, secType, /*IsSigned=*/false);
  Value *stride = builder.CreateMul(cstPIV, step);
  Value *repl = builder.CreateLShr(secInit, stride);

  return eliminate(secIV, repl, loop);
}

// Replace a secondary induction variable, `siv` with initial value `init`,
// step `step`, and type `type`. Here `type` is an integer type, and the updated
// value `siv'` is:
//
//     siv' = siv >>> step
//
// where `>>>` is the arithmetic right shift operator. `siv` can be computed
// from `civ`, the canonical induction variable of the loop as shown:
//
//     siv = init >>> step*civ
//
bool SecondaryIVElimination::tryToEliminateIntAShr(PHINode &secIV,
                                                   Constant *step, Loop &loop) {
  LLVM_DEBUG(dbgs() << "SecIVE:   Arithmetic shift right induction with step '"
                    << *step << "'\n");

  BasicBlock *ph = loop.getLoopPreheader();
  Value *secInit = secIV.getIncomingValueForBlock(ph);
  Type *secType = secIV.getType();
  PHINode *primIV = loop.getCanonicalInductionVariable();

  IRBuilder<> builder = getIRBuilder(loop);
  Value *cstPIV = builder.CreateIntCast(primIV, secType, /*IsSigned=*/false);
  Value *stride = builder.CreateMul(cstPIV, step);
  Value *repl = builder.CreateAShr(secInit, stride);

  return eliminate(secIV, repl, loop);
}

bool SecondaryIVElimination::tryToEliminate(PHINode &secIV, Loop &loop) {
  LLVM_DEBUG(dbgs() << "SecIVE:   Found secondary induction '" << getName(secIV)
                    << "'\n");

  // Don't need to check anything here because everything has been validated
  // in check(). The loop is in simplify form, so a preheader and unique latch
  // are guaranteed to exist.
  Value *upd = secIV.getIncomingValueForBlock(loop.getLoopLatch());
  BinaryOperator *binOp = cast<BinaryOperator>(upd);
  Constant *step = cast<Constant>(getNonMatchingOperand(*binOp, &secIV));
  switch (binOp->getOpcode()) {
  case BinaryOperator::Add:
    return tryToEliminateIntAdd(secIV, step, loop);
  case BinaryOperator::Sub:
    return tryToEliminateIntSub(secIV, step, loop);
  case BinaryOperator::Mul:
    return tryToEliminateIntMul(secIV, step, loop);
  case BinaryOperator::SDiv:
    return tryToEliminateIntDiv(secIV, step, /*isSigned=*/true, loop);
  case BinaryOperator::UDiv:
    return tryToEliminateIntDiv(secIV, step, /*isSigned=*/false, loop);
  case BinaryOperator::Shl:
    return tryToEliminateIntLShl(secIV, step, loop);
  case BinaryOperator::LShr:
    return tryToEliminateIntLShr(secIV, step, loop);
  case BinaryOperator::AShr:
    return tryToEliminateIntAShr(secIV, step, loop);
  case BinaryOperator::FAdd:
    return tryToEliminateFPAdd(secIV, step, loop);
  case BinaryOperator::FSub:
    return tryToEliminateFPSub(secIV, step, loop);
  case BinaryOperator::FMul:
    return tryToEliminateFPMul(secIV, step, loop);
  case BinaryOperator::FDiv:
    return tryToEliminateFPDiv(secIV, step, loop);
  default:
    llvm_unreachable("tryToEliminate: BinaryOperator not handled");
  }
}

bool SecondaryIVElimination::check(Loop &loop) {
  auto isSupportedBinOp = [](unsigned op) -> bool {
    switch (op) {
    case BinaryOperator::Add:
    case BinaryOperator::Sub:
    case BinaryOperator::Mul:
    case BinaryOperator::SDiv:
    case BinaryOperator::UDiv:
    case BinaryOperator::Shl:
    case BinaryOperator::LShr:
    case BinaryOperator::AShr:
    case BinaryOperator::FAdd:
    case BinaryOperator::FSub:
    case BinaryOperator::FMul:
    case BinaryOperator::FDiv:
      return true;
    default:
      return false;
    }
  };

  auto getBinOpName = [](unsigned op) -> StringRef {
    // clang-format off
    switch (op) {
    case BinaryOperator::Add: return "add";
    case BinaryOperator::Sub: return "sub";
    case BinaryOperator::Mul: return "imul";
    case BinaryOperator::SDiv: return "sdiv";
    case BinaryOperator::UDiv: return "udiv";
    case BinaryOperator::SRem: return "srem";
    case BinaryOperator::URem: return "urem";
    case BinaryOperator::Shl: return "lshl";
    case BinaryOperator::LShr: return "lshr";
    case BinaryOperator::AShr: return "ashr";
    case BinaryOperator::FAdd: return "fadd";
    case BinaryOperator::FSub: return "fsub";
    case BinaryOperator::FMul: return "fmul";
    case BinaryOperator::FDiv: return "fdiv";
    case BinaryOperator::FRem: return "frem";
    case BinaryOperator::And: return "and";
    case BinaryOperator::Or: return "or";
    case BinaryOperator::Xor: return "xor";
    }
    // clang-format on
    llvm_unreachable("getBinOpName: Binary opcode not handled");
  };

  // Check if one argument of the binary operator is the given induction
  // variable and the other is a constant.
  auto hasConstantStep = [](BinaryOperator &binOp, PHINode &iv) -> bool {
    if (Value *step = getNonMatchingOperand(binOp, &iv))
      return isa<Constant>(step);
    return false;
  };

  // We have to ensure that there are no non-phi instructions in the header.
  // This is strictly enforced for tapir loops just loop-spawning, but we have
  // to do so here as well. The code to compute the secondary IV from the
  // primary IV will be inserted in the detached block of the tapir loop. If
  // any uses of the secondary IV are present in the header, this will result
  // in an invalid module.
  BasicBlock *header = loop.getHeader();
  if (!isa<DetachInst>(header->getTerminator()))
    return complain(loop, DiagID::ErrTapirLoopBlockTerminator, "header",
                    "detach");

  // This is a sanity check in case a pass that ran before this broke the tapir
  // loop in some way. The code in this pass expects the detach instruction to
  // be present.
  for (Instruction &inst : *header)
    if (!isa<PHINode>(inst) && !isa<DetachInst>(inst))
      return complain(loop, DiagID::ErrTapirLoopHeaderInstNotPHI);

  // Th computation of the secondary IV as a function of the primary IV assumes
  // that the primary IV is canonical.
  PHINode *primIV = loop.getCanonicalInductionVariable();
  if (!primIV)
    return complain(loop, DiagID::ErrTapirLoopIVNotCanonical);

  // Strictly speaking, this pass only requires the loop to have a unique latch,
  // but it is safer to require the loop to be in simplify-form.
  if (!loop.isLoopSimplifyForm())
    return complain(loop, DiagID::ErrLoopNotSimplifyForm);

  BasicBlock *latch = loop.getLoopLatch();

  for (PHINode &iv : loop.getHeader()->phis()) {
    if (&iv == primIV)
      continue;

    // We only support eliminating certain forms of secondary inductions. If the
    // type is one that we do not support, don't bother checking anything else,
    // just bail.
    //
    // NOTE: In principle, pointer inductions in parallel loops can be handled,
    // and it might actually be interesting to do so. But we don't support that
    // at this time.
    Type *ivTy = iv.getType();
    if (!ivTy->isIntegerTy() && !ivTy->isFloatingPointTy())
      return complain(iv, DiagID::ErrSecondaryIVType, *ivTy);

    if (isUsedOutsideLoop(iv, loop, li))
      return complain(iv, DiagID::ErrTapirLoopIVUsedOutsideLoop);

    // We currently only support eliminating secondary inductions of the form
    //
    //     x = x OP c
    //
    // where OP must be a supported binary operator and c must be a compile-time
    // constant. In principle, c only needs to be a pure function, but there is
    // no way to reliably determine that at this time. But we probably ought to
    // support arithmetic expressions as the right-hand side of OP. It is not
    // clear, though, if that is actually relevant for us.
    Value *upd = iv.getIncomingValueForBlock(latch);
    BinaryOperator *binOp = dyn_cast<BinaryOperator>(upd);
    if (!binOp)
      return complain(iv, DiagID::ErrSecondaryIVNotBinOp, *upd);
    else if (!hasConstantStep(*binOp, iv))
      return complain(iv, DiagID::ErrSecondaryIVNonConstStep);

    unsigned opcode = binOp->getOpcode();
    if (!isSupportedBinOp(opcode))
      return complain(iv, DiagID::ErrSecondaryIVOperator, getBinOpName(opcode));

    // The only uses of the secondary IV in the loop latch must be the increment
    // that flows back to the loop header. If that is not the case, we may not
    // be able to safely eliminate the IV.
    SmallVector<Value *, 1> usesInLatch;
    for (Use &u : iv.uses())
      if (isUseInBlock(u, *latch))
        usesInLatch.push_back(u.getUser());

    if (usesInLatch.size() != 1)
      return complain(iv, DiagID::ErrTapirLoopIVUsesInLatch);
    else if (iv.getIncomingValueForBlock(latch) != usesInLatch[0])
      return complain(iv, DiagID::ErrTapirLoopIVNotUpdatedInLatch);
  }

  return true;
}

static SmallVector<PHINode *, 4> getSecIVs(Loop &loop) {
  SmallVector<PHINode *, 4> secIVs;
  PHINode *primIV = loop.getCanonicalInductionVariable();
  for (PHINode &iv : loop.getHeader()->phis())
    if (&iv != primIV)
      secIVs.push_back(&iv);
  return secIVs;
}

bool SecondaryIVElimination::run(Loop &loop) {
  bool changed = false;
  if (!isTapirLoop(loop) || getNumIndVars(loop) <= 1)
    return false;

  // If the loop must be transformed, and it cannot be for whatever reason, it
  // will cause issues later in the pipeline. In the worst case, the code will
  // be silently miscompiled. To avoid this, if something cannot be transformed,
  // fail immediately.
  if (!check(loop))
    exitOnError();

  LLVM_DEBUG(dbgs() << "SecIVE: Checking loop " << getName(loop) << "\n");
  SmallVector<PHINode *, 4> secIVs = getSecIVs(loop);
  for (PHINode *iv : secIVs)
    changed |= tryToEliminate(*iv, loop);

  return changed;
}

PreservedAnalyses
SecondaryIVEliminationPass::run(Loop &loop, LoopAnalysisManager &am,
                                LoopStandardAnalysisResults &ar,
                                LPMUpdater &updater) {
  LoopInfo &li = ar.LI;

  if (!SecondaryIVElimination(li).run(loop))
    return PreservedAnalyses::all();
  return getLoopPassPreservedAnalyses();
}
