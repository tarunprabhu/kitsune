//===- Diagnostics.h - Kitsune-specific diagnostics ------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unlike LLVM, Kitsune's middle-end passes may emit diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_FRONTEND_DIAGNOSTICS_H
#define KITSUNE_FRONTEND_DIAGNOSTICS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class DebugInfo;
class Function;
class Instruction;
class Loop;

/// ID's for the various diagnostics that can be emitted by Kitsune. These are
/// only those diagnostics emitted in Kitsune's middle-end. Unlike LLVM which
/// only emits diagnostics in the frontend, and occasionally the back, Kitsune
/// emits diagnostics in the middle-end. These use source-level information if
/// it is available (through debug info).
/// \ingroup kitsune
enum class DiagID {
#define GET_DIAG_ENUMS
#include "kitsune/Frontend/Diagnostics.inc"
};

namespace detail {

// Get the severity of the given diagnostic.
DiagnosticSeverity getSeverity(DiagID id);

// Get the message emitted by the given diagnostic. The message may be a format
// string that is compatible with llvm::formatv.
StringRef getMsg(DiagID id);

/// Emit a diagnostic.
raw_ostream &emitDiagnostic(raw_ostream &os, DiagnosticSeverity severity,
                            StringRef msg);

/// Emit a diagnostic for the given value. This will only have a noticeable
/// effect if the value is a function or an instruction. In all other cases, it
/// will be ignored.
raw_ostream &emitDiagnostic(raw_ostream &os, const Value &v,
                            DiagnosticSeverity severity, StringRef msg);

/// Emit a diagnostic for the given loop.
raw_ostream &emitDiagnostic(raw_ostream &os, const Loop &loop,
                            DiagnosticSeverity severity, StringRef msg);

/// Emit a diagnostic for the given instruction.
raw_ostream &emitDiagnostic(raw_ostream &os, const Instruction &inst,
                            DiagnosticSeverity severity, StringRef msg);

/// Emit a diagnostic for the given function.
raw_ostream &emitDiagnostic(raw_ostream &os, const Function &f,
                            DiagnosticSeverity severity, StringRef msg);

} // namespace detail

/// \addtogroup kitsune
/// @{

/// Emit a diagnostic to stderr.
/// \param id The diagnostic to emit
void emitDiagnostic(DiagID id);

/// Emit a diagnostic to stderr.
/// \param id The diagnostic to emit
template <typename... Args> void emitDiagnostic(DiagID id, Args &&...args) {
  detail::emitDiagnostic(errs(), detail::getSeverity(id),
                         formatv(detail::getMsg(id).data(), args...).str());
}

/// Emit a diagnostic to stderr.
/// \param e  The IR element to be associated with the diagnostic. This may be a
///           Function, Instruction, or Loop. If valid debug information is
///           associated with the instruction or loop, it will be emitted in the
///           diagnostic.
/// \param id The diagnostic to emit
template <typename IRElement, typename... Args>
void emitDiagnostic(const IRElement &e, DiagID id) {
  detail::emitDiagnostic(errs(), e, detail::getSeverity(id),
                         detail::getMsg(id));
}

/// Emit a diagnostic to stderr.
/// \param e  The IR element to be associated with the diagnostic. This may be a
///           Function, Instruction, or Loop. If valid debug information is
///           associated with the instruction or loop, it will be emitted in the
///           diagnostic.
/// \param id The diagnostic to emit
template <typename IRElement, typename... Args>
void emitDiagnostic(const IRElement &e, DiagID id, Args &&...args) {
  detail::emitDiagnostic(errs(), e, detail::getSeverity(id),
                         formatv(detail::getMsg(id).data(), args...).str());
}

/// @}

} // namespace llvm

#endif // KITSUNE_FRONTEND_DIAGNOSTICS_H
