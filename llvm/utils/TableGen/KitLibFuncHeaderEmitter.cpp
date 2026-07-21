//==- KitLibFuncHeaderEmitter.cpp - Generate header for library functions --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kit-libfunc-header"

using namespace llvm;

namespace {

class KitLibFuncEmitter {
private:
  const RecordKeeper &records;

private:
  raw_ostream &emitCTypeEnums(raw_ostream &os);
  raw_ostream &emitCTypes(raw_ostream &os);
  raw_ostream &emitLibFuncEnums(raw_ostream &os);
  raw_ostream &emitLibFuncs(raw_ostream &os);

public:
  KitLibFuncEmitter(const RecordKeeper &records);

  void run(raw_ostream &os);
};

} // namespace

raw_ostream &KitLibFuncEmitter::emitCTypeEnums(raw_ostream &os) {
  os << "#ifdef GET_CTYPE_ENUMS\n";
  os << "#undef GET_CTYPE_ENUMS\n";
  os << "\n";

  unsigned val = 1;
  for (const Record *ctype : records.getAllDerivedDefinitions("CType"))
    os << ctype->getName() << " = " << val++ << ",\n";

  os << "\n";
  os << "#endif // GET_CTYPE_ENUMS\n";

  return os;
}

raw_ostream &KitLibFuncEmitter::emitCTypes(raw_ostream &os) {
  os << "#ifdef GET_CTYPES\n";
  os << "#undef GET_CTYPES\n";

  os << "\n";
  os << "#ifndef CTYPE\n";
  os << "#define CTYPE(NAME, CTYPE)\n";
  os << "#endif // CTYPE\n";
  os << "\n";

  for (const Record *ctype : records.getAllDerivedDefinitions("CType"))
    os << "CTYPE(" << ctype->getName() << ", \""
       << ctype->getValueAsString("Name") << "\")\n";

  os << "\n";
  os << "#undef CTYPE\n";

  os << "\n";
  os << "#endif // GET_CTYPES";

  return os;
}

raw_ostream &KitLibFuncEmitter::emitLibFuncEnums(raw_ostream &os) {
  os << "#ifdef GET_LIBFUNC_ENUMS\n";
  os << "#undef GET_LIBFUNC_ENUMS\n";

  unsigned val = 1;
  for (const Record *libFunc : records.getAllDerivedDefinitions("KitFunc"))
    os << libFunc->getName() << " = " << val++ << ",\n";

  os << "\n";
  os << "#endif // GET_LIBFUNC_ENUMS\n";

  return os;
}

raw_ostream &KitLibFuncEmitter::emitLibFuncs(raw_ostream &os) {
  os << "#ifdef GET_LIBFUNCS\n";
  os << "#undef GET_LIBFUNCS\n";

  os << "\n";
  os << "#ifndef LIBFUNC\n";
  os << "#define LIBFUNC(NAME, LINKAGE_NAME, ...)\n";
  os << "#endif // LIBFUNC\n";
  os << "\n";

  for (const Record *func : records.getAllDerivedDefinitions("KitFunc")) {
    os << "LIBFUNC(";
    os << func->getName() << ", ";
    os << "\"" << func->getValueAsString("Name") << "\", ";
    os << func->getValueAsDef("Ret")->getName();
    for (const Record *param : func->getValueAsListOfDefs("Params"))
      os << ", " << param->getName();
    os << ")\n";
  }

  os << "\n";
  os << "#undef LIBFUNC\n";

  os << "\n";
  os << "#endif // GET_LIBFUNCS\n";

  return os;
}

void KitLibFuncEmitter::run(raw_ostream &os) {
  emitCTypeEnums(os) << "\n\n";
  emitCTypes(os) << "\n\n";
  emitLibFuncEnums(os) << "\n\n";
  emitLibFuncs(os);
}

KitLibFuncEmitter::KitLibFuncEmitter(const RecordKeeper &records)
    : records(records) {}

static TableGen::Emitter::OptClass<KitLibFuncEmitter>
    X("gen-kit-libfunc-header",
      "Generate header for Kitsune-specific loop attributes");
