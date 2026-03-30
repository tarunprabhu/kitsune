//===- DiagnosticsInternal.h - Implementation of diagnostics ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal implementation header of Kitsune's diagnostics. This is not intended
// to be used directly, expect in certain circumstances.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_DIAGNOSTICS_INTERNAL_H
#define KITSUNE_SUPPORT_DIAGNOSTICS_INTERNAL_H

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class Function;
class Instruction;
class Loop;
class Value;
enum class DiagID : unsigned;

namespace detail {

/// Get the severity of the given diagnostic.
DiagnosticSeverity getSeverity(DiagID id);

/// Get the message emitted by the given diagnostic. The message may be a
/// format string that is compatible with llvm::formatv.
StringRef getMsg(DiagID id);

/// Emit a diagnostic.
raw_ostream &emitDiagnostic(raw_ostream &os, DiagnosticSeverity severity,
                            StringRef msg);

/// Emit a diagnostic for the given value. This will only have a noticeable
/// effect if the value is a function or an instruction. In all other cases,
/// it will be ignored.
raw_ostream &emitDiagnostic(raw_ostream &os, const Value &v,
                            DiagnosticSeverity severity, StringRef msg);

/// Emit a diagnostic for the given argument.
raw_ostream &emitDiagnostic(raw_ostream &os, const Argument &a,
                            DiagnosticSeverity severity, StringRef msg);

/// Emit a diagnostic for the given function.
raw_ostream &emitDiagnostic(raw_ostream &os, const Function &f,
                            DiagnosticSeverity severity, StringRef msg);

/// Emit a diagnostic for the given global variable.
raw_ostream &emitDiagnostic(raw_ostream &os, const GlobalVariable &g,
                            DiagnosticSeverity severity, StringRef msg);

/// Emit a diagnostic for the given instruction.
raw_ostream &emitDiagnostic(raw_ostream &os, const Instruction &inst,
                            DiagnosticSeverity severity, StringRef msg);

/// Emit a diagnostic for the given loop.
raw_ostream &emitDiagnostic(raw_ostream &os, const Loop &loop,
                            DiagnosticSeverity severity, StringRef msg);

} // namespace detail

} // namespace llvm

#endif // KITSUNE_SUPPORT_DIAGNOSTICS_INTERNAL_H
