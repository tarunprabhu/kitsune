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

#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/DIUtils.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/ValueUtils.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;

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

static raw_ostream &emitPrefix(raw_ostream &os, StringRef name) {
  if (name.size()) {
    bold(os) << name;
    normal(os) << ": ";
  }
  return os;
}

static raw_ostream &emitMsg(raw_ostream &os, StringRef msg) {
  if (msg.size()) {
    bold(os) << msg;
    normal(os) << "\n";
  }
  return os;
}

raw_ostream &llvm::detail::emitDiagnostic(raw_ostream &os,
                                          DiagnosticSeverity severity,
                                          StringRef msg) {
  emitLabel(os, severity);
  emitMsg(os, msg);

  return os;
}

raw_ostream &llvm::detail::emitDiagnostic(raw_ostream &os, const Argument &a,
                                          DiagnosticSeverity severity,
                                          StringRef msg) {
  emitLabel(os, severity);
  emitMsg(os, msg);
  llvm::emitDiagnosticTo(os, DiagID::NoteFromArgument, getName(a),
                         getName(*a.getParent()));
  return os;
}

raw_ostream &llvm::detail::emitDiagnostic(raw_ostream &os, const Function &f,
                                          DiagnosticSeverity severity,
                                          StringRef msg) {
  emitLabel(os, severity);
  emitMsg(os, msg);
  llvm::emitDiagnosticTo(os, DiagID::NoteFromFunction, getName(f));

  return os;
}

raw_ostream &llvm::detail::emitDiagnostic(raw_ostream &os,
                                          const GlobalVariable &g,
                                          DiagnosticSeverity severity,
                                          StringRef msg) {
  emitLabel(os, severity);
  emitMsg(os, msg);
  llvm::emitDiagnosticTo(os, DiagID::NoteFromGlobalVariable, getName(g));

  return os;
}

raw_ostream &llvm::detail::emitDiagnostic(raw_ostream &os,
                                          const Instruction &inst,
                                          DiagnosticSeverity severity,
                                          StringRef msg) {
  std::string loc = toString(inst.getStableDebugLoc());
  emitLabel(os, severity);
  emitPrefix(os, loc);
  emitMsg(os, msg);
  if (loc.empty() && severity != DiagnosticSeverity::DS_Note) {
    if (const BasicBlock *bb = inst.getParent())
      if (bb->hasName())
        emitDiagnosticTo(os, DiagID::NoteFromBasicBlock, getName(*bb));
    if (const Function *f = inst.getFunction())
      emitDiagnosticTo(os, DiagID::NoteFromFunction, getName(*f));
  }
  return os;
}

raw_ostream &llvm::detail::emitDiagnostic(raw_ostream &os, const Loop &loop,
                                          DiagnosticSeverity severity,
                                          StringRef msg) {
  std::string loc = toString(loop.getStartLoc());
  emitLabel(os, severity);
  emitPrefix(os, loc);
  emitMsg(os, msg);
  if (loc.empty() && severity != DiagnosticSeverity::DS_Note) {
    std::string name = getName(loop);
    if (name.size())
      emitDiagnosticTo(os, DiagID::NoteFromLoop, name);
    if (const Function *f = getFunction(loop))
      emitDiagnosticTo(os, DiagID::NoteFromFunction, getName(*f));
  }
  return os;
}

raw_ostream &llvm::detail::emitDiagnostic(raw_ostream &os, const Value &v,
                                          DiagnosticSeverity severity,
                                          StringRef msg) {
  if (const auto *a = dyn_cast<Argument>(&v))
    return emitDiagnostic(os, *a, severity, msg);
  else if (const auto *f = dyn_cast<Function>(&v))
    return emitDiagnostic(os, *f, severity, msg);
  else if (const auto *g = dyn_cast<GlobalVariable>(&v))
    return emitDiagnostic(os, *g, severity, msg);
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
#include "kitsune/Core/Diagnostics.inc"
  }
  llvm_unreachable("getSeverity: DiagID not handled");
}

StringRef llvm::detail::getMsg(DiagID id) {
  switch (id) {
#define GET_DIAGS
#define DIAG(NAME, SEVERITY, MSG)                                              \
  case DiagID::NAME:                                                           \
    return MSG;
#include "kitsune/Core/Diagnostics.inc"
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

namespace llvm {
namespace detail {

// Emit a diagnostic indicating that the pass \p requires needed by the pass
// \p pass was not run. This is only intended to be used in PassUtilsInternal.h.
void emitPassNotRunDiagnostic(StringRef pass, StringRef reqd) {
  DiagID id = DiagID::ErrRequiredPassNotRun;
  emitDiagnostic(errs(), getSeverity(id),
                 formatv(getMsg(id).data(), pass, reqd).str());
}

// Emit a diagnostic indicating stating that some required passes were not run
// and exit with a system-dependent error code. This function is only intended
// to be used in PassUtilsInternal.h
void emitFatalPassesNotRunDiagnostic() {
  // Despite the name of the function, we don't actually emit a diagnostic
  // here. Maybe we will at some point.
  exitOnError();
}

} // namespace detail
} // namespace llvm
