//===- Diagnostics.cpp - Kitsune-specific diagnostics ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to display diagnostics with Kitsune. Unlike LLVM, Kitsune's
// middle-end passes may emit diagnostics.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Frontend/Diagnostics.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;

static bool hasLoc(DebugLoc dbgLoc) {
  if (dbgLoc)
    if (const auto *diScope = dyn_cast<DIScope>(dbgLoc.getScope()))
      return diScope->getFilename().size();
  return false;
}

static raw_ostream &bold(raw_ostream &os) {
  if (WithColor(os).colorsEnabled())
    os.changeColor(raw_ostream::SAVEDCOLOR, /*bold=*/true);
  return os;
}

static raw_ostream &normal(raw_ostream &os) {
  if (WithColor(os).colorsEnabled())
    os.resetColor();
  return os;
}

static raw_ostream &emitLabel(raw_ostream &os, DiagnosticSeverity severity) {
  switch (severity) {
  case DiagnosticSeverity::DS_Error:
    return WithColor::error(os);
  case DiagnosticSeverity::DS_Warning:
    return WithColor::warning(os);
  case DiagnosticSeverity::DS_Remark:
    return WithColor::remark(os);
  case DiagnosticSeverity::DS_Note:
    return WithColor::note(os);
  }
  llvm_unreachable("emitLabel: Diagnostic severity not handled");
}

static raw_ostream &emitPrefix(raw_ostream &os, const DebugLoc &dbgLoc) {
  bold(os);
  os << cast<DIScope>(dbgLoc.getScope())->getFilename();
  if (unsigned line = dbgLoc.getLine())
    os << ":" << line;
  if (unsigned col = dbgLoc.getCol())
    os << ":" << col;
  os << ": ";
  normal(os);

  return os;
}

static raw_ostream &emitPrefix(raw_ostream &os, const Function &f) {
  bold(os);
  os << "in function '" << f.getName() << "': ";
  normal(os);

  return os;
}

static raw_ostream &emitMsg(raw_ostream &os, StringRef msg) {
  bold(os);
  os << msg << "\n";
  normal(os);

  return os;
}

template <typename T>
static raw_ostream &emitDiagnostic(raw_ostream &os, DiagnosticSeverity severity,
                                   StringRef msg, const T &dbgOrFunc) {
  emitLabel(os, severity);
  emitPrefix(os, dbgOrFunc);
  emitMsg(os, msg);

  return os;
}

raw_ostream &llvm::detail::emitDiagnostic(raw_ostream &os,
                                          DiagnosticSeverity severity,
                                          StringRef msg) {
  emitLabel(os, severity);
  emitMsg(os, msg);

  return os;
}

raw_ostream &llvm::detail::emitDiagnostic(raw_ostream &os, const Function &f,
                                          DiagnosticSeverity severity,
                                          StringRef msg) {
  emitLabel(os, severity);
  emitPrefix(os, f);
  emitMsg(os, msg);

  return os;
}

raw_ostream &llvm::detail::emitDiagnostic(raw_ostream &os,
                                          const Instruction &inst,
                                          DiagnosticSeverity severity,
                                          StringRef msg) {
  DebugLoc dbgLoc = inst.getStableDebugLoc();
  if (hasLoc(dbgLoc))
    ::emitDiagnostic(os, severity, msg, dbgLoc);
  else if (const Function *f = inst.getFunction())
    ::emitDiagnostic(os, severity, msg, *f);
  else
    emitDiagnostic(os, severity, msg);
  return os;
}

raw_ostream &llvm::detail::emitDiagnostic(raw_ostream &os, const Loop &loop,
                                          DiagnosticSeverity severity,
                                          StringRef msg) {
  auto getDebugLoc = [](const Loop &loop) -> DebugLoc {
    if (BasicBlock *latch = loop.getLoopLatch())
      if (DebugLoc dbg = latch->getTerminator()->getStableDebugLoc())
        return dbg;
    return DebugLoc();
  };

  auto getFunction = [](const Loop &loop) -> const Function * {
    return loop.getHeader()->getParent();
  };

  DebugLoc dbgLoc = getDebugLoc(loop);
  if (hasLoc(dbgLoc))
    ::emitDiagnostic(os, severity, msg, dbgLoc);
  else if (const Function *f = getFunction(loop))
    ::emitDiagnostic(os, severity, msg, *f);
  else
    emitDiagnostic(os, severity, msg);
  return os;
}

raw_ostream &llvm::detail::emitDiagnostic(raw_ostream &os, const Value &v,
                                          DiagnosticSeverity severity,
                                          StringRef msg) {
  if (const auto *f = dyn_cast<Function>(&v))
    return emitDiagnostic(os, *f, severity, msg);
  else if (const auto *inst = dyn_cast<Instruction>(&v))
    return emitDiagnostic(os, *inst, severity, msg);
  else
    return emitDiagnostic(os, severity, msg);
}

DiagnosticSeverity llvm::detail::getSeverity(DiagID id) {
  switch (id) {
#define GET_DIAGS
#define DIAG(NAME, SEVERITY, MSG)                                              \
  case DiagID::NAME:                                                           \
    return SEVERITY;
#include "kitsune/Frontend/Diagnostics.inc"
  }
  llvm_unreachable("getSeverity: DiagID not handled");
}

StringRef llvm::detail::getMsg(DiagID id) {
  switch (id) {
#define GET_DIAGS
#define DIAG(NAME, SEVERITY, MSG)                                              \
  case DiagID::NAME:                                                           \
    return MSG;
#include "kitsune/Frontend/Diagnostics.inc"
  }
  llvm_unreachable("getMsg: DiagID not handled");
}

bool llvm::isError(DiagID id) {
  return detail::getSeverity(id) == DiagnosticSeverity::DS_Error;
}

bool llvm::isWarning(DiagID id) {
  return detail::getSeverity(id) == DiagnosticSeverity::DS_Warning;
}

bool llvm::isRemark(DiagID id) {
  return detail::getSeverity(id) == DiagnosticSeverity::DS_Remark;
}

bool llvm::isNote(DiagID id) {
  return detail::getSeverity(id) == DiagnosticSeverity::DS_Note;
}

void llvm::emitDiagnostic(DiagID id) {
  detail::emitDiagnostic(errs(), detail::getSeverity(id), detail::getMsg(id));
}
