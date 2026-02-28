//===- KitsuneDiagEmitter.cpp - Generate Kitsune-specific diagnostics -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kitsune-diags-emitter"

using namespace llvm;

static raw_ostream &line(raw_ostream &os, StringRef s = "") {
  os << s << "\n";
  return os;
}

static StringRef getSeverity(const Record &r) {
  StringRef sev = StringSwitch<StringRef>(r.getName())
                      .Case("SErr", "DiagnosticSeverity::DS_Error")
                      .Case("SWarn", "DiagnosticSeverity::DS_Warning")
                      .Case("SRemark", "DiagnosticSeverity::DS_Remark")
                      .Case("SNote", "DiagnosticSeverity::DS_Note")
                      .Default("");
  if (sev.empty())
    PrintFatalError(r.getLoc(), "Severity not handled by emitter");
  return sev;
}

class DiagsEmitter {
private:
  SmallVector<const Record *, 16> records;

private:
  void emitDiags(raw_ostream &os) {
    line(os, "#ifdef GET_DIAGS");
    line(os, "#undef GET_DIAGS");
    line(os);
    line(os, "#ifndef DIAG");
    line(os, "#define DIAG(NAME, SEVERITY, MSG)");
    line(os, "#endif // DIAG");
    line(os);

    for (const Record *r : records) {
      StringRef name = r->getName();
      StringRef msg = r->getValueAsString("Msg");
      StringRef severity = getSeverity(*r->getValueAsDef("Sev"));

      os << "DIAG(" << name << ", " << severity << ", \"" << msg
         << "\")\n";
    }

    line(os);
    line(os, "#undef DIAG");
    line(os, "#endif // GET_DIAGS");
  }

  void emitDiagEnums(raw_ostream &os) {
    line(os, "#ifdef GET_DIAG_ENUMS");
    line(os, "#undef GET_DIAG_ENUMS");
    line(os);

    // The first diagnostic id value is 1.
    unsigned val = 1;
    for (const Record *r : records) {
      os << r->getName() << " = " << val << ",\n";
      ++val;
    }

    line(os);
    line(os, "#endif // GET_DIAG_ENUMS");
  }

public:
  DiagsEmitter(const RecordKeeper &recordKeeper) {
    for (const Record *r : recordKeeper.getAllDerivedDefinitions("Diag"))
      records.push_back(r);

    std::sort(records.begin(), records.end(),
              [](const Record *l, const Record *r) {
                return l->getName() < r->getName();
              });
  }

  void run(raw_ostream &os) {
    emitDiags(os);
    line(os);
    line(os, "// ------------------------------------------------------------");
    line(os);
    emitDiagEnums(os);
  }
};

static TableGen::Emitter::OptClass<DiagsEmitter>
    X("gen-kitsune-diags", "Generate Kitsune-specific diagnostics");
