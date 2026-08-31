//==- KitIntrLibFuncMapEmitter.cpp - Generate header for library functions -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit macros to generate a map from Kitsune's intrinsics to runtime functions
// that will be used for lowering.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kit-intr-libfunc-map"

using namespace llvm;

namespace {

class KitIntrLibFuncMapEmitter {
private:
  const RecordKeeper &records;

private:
  raw_ostream &emitMapEntry(const Record &rt, raw_ostream &os);
  raw_ostream &emitIntrLibFuncMap(raw_ostream &os);
  raw_ostream &emitIntr(const Record &intr, raw_ostream &os);
  raw_ostream &emitIntrs(raw_ostream &os);

public:
  KitIntrLibFuncMapEmitter(const RecordKeeper &records) : records(records) {}

  void run(raw_ostream &os);
};

} // namespace

static std::string getLowerMode(const Record &intr) {
  std::string buf;
  raw_string_ostream os(buf);

  // This is the name of the scoped enum in kitsune/Core/IntrinsicUtils.h that
  // is used in the lowering mode field.
  os << "KitIntrLowerMode::";

  // If an RTSpec has been set, then the lowering mode is Runtime. Otherwise,
  // use the value of the LowerMode field. The value is guaranteed to be set.
  // The name will always be prefixed with Lower, so that must be stripped.
  std::vector<const Record *> rtSpec = intr.getValueAsListOfDefs("RTSpec");
  if (!rtSpec.empty())
    os << "Runtime";
  else
    os << intr.getValueAsDef("LowerMode")->getName().substr(5);
  os.flush();

  return buf;
}

raw_ostream &KitIntrLibFuncMapEmitter::emitIntr(const Record &intr,
                                                raw_ostream &os) {
  os << "INTR(";
  // The name is always prefixed with int_. This should be removed to get the
  // corresponding enum name.
  os << intr.getName().substr(4) << ", ";
  os << getLowerMode(intr) << ", ";
  os << intr.getValueAsBit("AllowParamCast") << ", ";
  os << intr.getValueAsBit("AllowReturnCast");
  os << ")";

  return os;
}

raw_ostream &KitIntrLibFuncMapEmitter::emitIntrs(raw_ostream &os) {
  os << "#ifdef GET_INTR_LOWERING_SPEC\n";
  os << "#undef GET_INTR_LOWERING_SPEC\n";
  os << "\n";

  os << "#ifndef INTR\n";
  os << "#define INTR(NAME, CUSTOM_LOWER, ALLOW_PARAM_CAST, "
        "ALLOW_RETURN_CAST)\n";
  os << "#endif // INTR\n";
  os << "\n";

  for (const Record *intr : records.getAllDerivedDefinitions("KitIntrinsic"))
    emitIntr(*intr, os) << "\n";

  os << "\n";
  os << "#undef INTR\n";

  os << "\n";
  os << "#endif // GET_INTR_LOWERING_SPEC\n";

  return os;
}

raw_ostream &KitIntrLibFuncMapEmitter::emitMapEntry(const Record &rt,
                                                    raw_ostream &os) {
  os << "{";
  os << "TTID::" << rt.getValueAsDef("TT")->getName();
  os << ", ";
  os << "KitFunc::" << rt.getValueAsDef("Func")->getName();
  os << "}";

  return os;
}

raw_ostream &KitIntrLibFuncMapEmitter::emitIntrLibFuncMap(raw_ostream &os) {
  os << "#ifdef GET_INTR_LIBFUNC_MAP\n";
  os << "#undef GET_INTR_LIBFUNC_MAP\n";
  os << "\n";

  os << "#define INTR_LIBFUNC_MAP { \\\n";
  for (const Record *intr : records.getAllDerivedDefinitions("KitIntrinsic")) {
    // The name is always prefixed with int_. This should be removed to get the
    // corresponding enum name.
    os << "  { Intrinsic::" << intr->getName().substr(4) << ", {";
    std::vector<const Record *> rtSpec = intr->getValueAsListOfDefs("RTSpec");
    if (rtSpec.size()) {
      emitMapEntry(*rtSpec[0], os);
      for (unsigned i = 1; i < rtSpec.size(); ++i) {
        os << ", ";
        emitMapEntry(*rtSpec[i], os);
      }
    }
    os << "} }, \\\n";
  }
  os << "}\n";

  os << "\n";
  os << "#endif // GET_INTR_LIBFUNC_MAP\n";

  return os;
}

void KitIntrLibFuncMapEmitter::run(raw_ostream &os) {
  for (const Record *intr : records.getAllDerivedDefinitions("KitIntrinsic")) {
    std::vector<const Record *> rtSpec = intr->getValueAsListOfDefs("RTSpec");
    StringRef lowerMode = intr->getValueAsDef("LowerMode")->getName();
    if (!rtSpec.empty() && lowerMode != "LowerUnspecified")
      PrintFatalError(
          intr->getLoc(),
          "If an RTSpec is provided, the LowerMode must be unspecfied");
  }

  emitIntrLibFuncMap(os);
  emitIntrs(os);
}

static TableGen::Emitter::OptClass<KitIntrLibFuncMapEmitter>
    X("gen-kit-intr-libfunc-map",
      "Generate maps between Kitsune's intrinsics and library functions");
