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

#ifndef KITSUNE_SUPPORT_DIAGNOSTICS_H
#define KITSUNE_SUPPORT_DIAGNOSTICS_H

#include "kitsune/Support/DiagnosticsInternal.h"
#include "kitsune/Support/OstreamUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class DebugInfo;
class Function;
class Instruction;
class Loop;

/// \addtogroup kitsune
/// @{

/// Wrapper class around diagnostic message strings.
class DiagMessage {
public:
  static constexpr StringRef errTTEmbBC =
      "Tapir target does not generate embedded bitcode";
};

/// ID's for the various diagnostics that can be emitted by Kitsune. These are
/// only those diagnostics emitted in Kitsune's middle-end. Unlike LLVM which
/// only emits diagnostics in the frontend, and occasionally the back, Kitsune
/// emits diagnostics in the middle-end. These use source-level information if
/// it is available (through debug info).
/// \ingroup kitsune
enum class DiagID : unsigned {
#define GET_DIAG_ENUMS
#include "kitsune/Support/Diagnostics.inc"
};

/// Is the diagnostic an error.
bool isError(DiagID id);

/// Is the diagnostic a warning.
bool isWarning(DiagID id);

/// Is the diagnostic a remark.
bool isRemark(DiagID id);

/// Is the diagnostic a note.
bool isNote(DiagID id);

/// Emit a diagnostic to stderr.
/// \param id The diagnostic to emit
void emitDiagnostic(DiagID id);

/// Emit a diagnostic to the given output stream.
/// \param os The output stream.
/// \param e  The IR element to be associated with the diagnostic. If valid
///           debug information is associated with the IR element, it may be
///           used in the emitted diagnostic. This element may be an attribute
///           kind
/// \param id The diagnostic to emit
/// \param args Additional arguments required by \p id.
template <typename IRElement, typename... Args>
void emitDiagnosticTo(raw_ostream &os, const IRElement &e, DiagID id,
                      Args &&...args) {
  detail::emitDiagnostic(os, e, detail::getSeverity(id),
                         formatv(detail::getMsg(id).data(), args...).str());
}

/// Emit a diagnostic to the given output stream.
/// \param os The output stream.
/// \param id The diagnostic to emit
/// \param args Additional arguments required by \p id.
template <typename... Args>
void emitDiagnosticTo(raw_ostream &os, DiagID id, Args &&...args) {
  detail::emitDiagnostic(os, detail::getSeverity(id),
                         formatv(detail::getMsg(id).data(), args...).str());
}

/// Emit a diagnostic to stderr.
/// \param id The diagnostic to emit
/// \param args Additional arguments required by \p id
template <typename... Args> void emitDiagnostic(DiagID id, Args &&...args) {
  emitDiagnosticTo(errs(), id, args...);
}

/// Emit a diagnostic to stderr.
/// \param e  The IR element to be associated with the diagnostic. If valid
///           debug information is associated with the IR element, it may be
///           used in the emitted diagnostic. The element may be an attribute
///           kind
/// \param id The diagnostic to emit
/// \param args Additional arguments required by \p id.
template <typename IRElement, typename... Args>
void emitDiagnostic(const IRElement &e, DiagID id, Args &&...args) {
  emitDiagnosticTo(errs(), e, id, args...);
}

/// Create an LLVM Error for a Kitsune-specific diagnostic.
template <typename... Args> Error createDiagError(DiagID id, Args &&...args) {
  return createStringError(formatv(detail::getMsg(id).data(), args...).str());
}

/// @}

} // namespace llvm

#endif // KITSUNE_SUPPORT_DIAGNOSTICS_H
