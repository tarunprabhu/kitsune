//===- KitsuneDiagEmitter.cpp - Generate Kitsune-specific diagnostics -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kitsune-diags-emitter"

using namespace llvm;

static raw_ostream &line(raw_ostream &os, StringRef s1 = "",
                         StringRef s2 = "") {
  os << s1;
  if (s2.size())
    os << " " << s2;
  os << "\n";

  return os;
}

static StringRef getMacroName(StringRef sev) {
  return StringSwitch<StringRef>(sev)
      .Case("Error", "DIAG_ERROR")
      .Case("Warning", "DIAG_WARNING")
      .Case("Remark", "DIAG_REMARK")
      .Case("Note", "DIAG_NOTE");
}

static StringRef getSeverity(StringRef sev) {
  return StringSwitch<StringRef>(sev)
      .Case("Error", "DiagnosticSeverity::DS_Error")
      .Case("Warning", "DiagnosticSeverity::DS_Warning")
      .Case("Remark", "DiagnosticSeverity::DS_Remark")
      .Case("Note", "DiagnosticSeverity::DS_Note");
}

class DiagsEmitter {
private:
  SetVector<StringRef> classes;
  const RecordKeeper &recordKeeper;

private:
  void emitDiags(raw_ostream &os) {
    line(os, "#ifdef GET_DIAGS");
    line(os, "#undef GET_DIAGS");
    line(os);
    line(os, "#ifndef DIAG");
    line(os, "#define DIAG(NAME, SEVERITY, MSG)");
    line(os, "#endif // DIAG");
    line(os);

    for (StringRef klass : classes) {
      StringRef macro = getMacroName(klass);
      StringRef severity = getSeverity(klass);

      line(os, "#ifndef", macro);
      os << "#define " << macro << "(NAME, MSG) \\\n";
      os << "    DIAG(NAME, " << severity << ", MSG)\n";
      line(os, "#endif //", macro);
      line(os);

      for (const Record *r : recordKeeper.getAllDerivedDefinitions(klass)) {
        StringRef name = r->getName();
        StringRef msg = r->getValueAsString("Msg");
        os << macro << "(" << name << ", \"" << msg << "\")\n";
      }

      line(os);
      line(os, "#undef", macro);
      line(os);
    }

    line(os, "#undef DIAG");
    line(os, "#endif // GET_DIAGS");
  }

  void emitDiagEnums(raw_ostream &os) {
    line(os, "#ifdef GET_DIAG_ENUMS");
    line(os, "#undef GET_DIAG_ENUMS");
    line(os);

    // The first diagnostic id value is 1.
    unsigned val = 1;
    for (StringRef klass : classes) {
      for (const Record *r : recordKeeper.getAllDerivedDefinitions(klass)) {
        os << r->getName() << " = " << val << ",\n";
        ++val;
      }
    }

    line(os);
    line(os, "#endif // GET_DIAG_ENUMS");
  }

public:
  DiagsEmitter(const RecordKeeper &recordKeeper) : recordKeeper(recordKeeper) {
    for (const Record *r : recordKeeper.getAllDerivedDefinitions("Diag")) {
      const Record *sev = r->getValueAsDef("Sev");
      StringRef klass = StringSwitch<StringRef>(sev->getName())
                            .Case("SErr", "Error")
                            .Case("SWarn", "Warning")
                            .Case("SRemark", "Remark")
                            .Case("SNote", "Note")
                            .Default("");
      if (klass.empty())
        PrintFatalError(sev->getLoc(), "Severity not handled by emitter");
      classes.insert(klass);
    }
  }

  void run(raw_ostream &os) {
    emitDiags(os);
    line(os);
    emitDiagEnums(os);
  }
};

static TableGen::Emitter::OptClass<DiagsEmitter>
    X("gen-kitsune-diags", "Generate Kitsune-specific diagnostics");
