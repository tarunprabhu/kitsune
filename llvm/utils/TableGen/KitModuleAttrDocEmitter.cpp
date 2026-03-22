//===- KitModuleAttrDocEmitter.cpp - Generate docs for module attributes --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KitAttrCommon.h"
#include "KitAttrDocEmitter.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kit-module-attr-doc"

using namespace llvm;

namespace {

class KitModuleAttrDocEmitter : public KitAttrDocEmitter {
protected:
  virtual StringRef getEnumName() const override;
  virtual StringRef getAttrBase() const override;
  virtual StringRef getLabelPrefix() const override;
  virtual StringRef getIRNamePrefix(const Record &attr) const override;

  virtual void emitAttrArgs(raw_ostream &os, const Record &attr) override;

public:
  KitModuleAttrDocEmitter(const RecordKeeper &records);
  virtual ~KitModuleAttrDocEmitter() = default;
};

} // namespace

StringRef KitModuleAttrDocEmitter::getEnumName() const {
  return "ModuleAttrKind";
}

StringRef KitModuleAttrDocEmitter::getAttrBase() const { return "ModuleAttr"; }

StringRef KitModuleAttrDocEmitter::getLabelPrefix() const { return "module"; }

StringRef KitModuleAttrDocEmitter::getIRNamePrefix(const Record &attr) const {
  return "kit.module.";
}

static raw_ostream &pad(raw_ostream &os, StringRef s, size_t w) {
  size_t l = s.size();
  size_t lp = (w - l) / 2;
  size_t rp = w - l - lp;
  for (unsigned i = 0; i < lp; ++i)
    os << " ";
  os << s;
  for (unsigned i = 0; i < rp; ++i)
    os << " ";
  return os;
}

static raw_ostream &line(raw_ostream &os, StringRef s1, size_t w1, StringRef s2,
                         size_t w2, StringRef s3, size_t w3) {
  os << "| ";
  pad(os, s1, w1) << " | ";
  pad(os, s2, w2) << " | ";
  pad(os, s3, w3);
  os << " |\n";

  return os;
}

static raw_ostream &sep(raw_ostream &os, char c, StringRef s1, StringRef s2,
                        StringRef s3) {
  os << "+" << c << s1;
  os << c << "+" << c << s2;
  os << c << "+" << c << s3;
  os << c << "+";
  os << "\n";

  return os;
}

static raw_ostream &sepSpan(raw_ostream &os, char c, size_t w1, StringRef s2,
                            StringRef s3) {
  os << "| ";
  pad(os, "", w1) << " ";
  os << "+" << c << s2 << c;
  os << "+" << c << s3 << c;
  os << "+\n";

  return os;
}

void KitModuleAttrDocEmitter::emitAttrArgs(raw_ostream &os,
                                           const Record &attr) {
  std::vector<std::string> types, names;
  for (const Record *value : attr.getValueAsListOfDefs("Values")) {
    StringRef type = getTypeName(*value);
    StringRef name = value->getValueAsString("ValueName");

    types.push_back(quote(type, "``"));
    names.push_back(quote(name, "``"));
  }

  std::string enm = getEnum(attr);
  std::string hdr1 = "Enum";
  size_t w1 = std::max(hdr1.size(), enm.size());
  std::string dash1 = std::string(w1, '-');
  std::string eq1 = std::string(w1, '=');

  std::string hdr2 = "Value Types";
  size_t w2 = hdr2.size();
  for (StringRef type : types)
    w2 = std::max(w2, type.size());
  std::string dash2 = std::string(w2, '-');
  std::string eq2 = std::string(w2, '=');

  std::string hdr3 = "Value Names";
  size_t w3 = hdr3.size();
  for (StringRef name : names)
    w3 = std::max(w3, name.size());
  std::string dash3 = std::string(w3, '-');
  std::string eq3 = std::string(w3, '=');

  sep(os, '-', dash1, dash2, dash3);
  line(os, hdr1, w1, hdr2, w2, hdr3, w3);
  sep(os, '=', eq1, eq2, eq3);
  if (types.size() > 0) {
    line(os, enm, w1, types[0], w2, names[0], w3);
    for (size_t i = 1; i < types.size(); ++i) {
      sepSpan(os, '-', w1, dash2, dash3);
      line(os, "", w1, types[i], w2, names[i], w3);
    }
  } else {
    line(os, enm, w1, "", w2, "", w3);
  }
  sep(os, '-', dash1, dash2, dash3);
  os << "\n";
}

KitModuleAttrDocEmitter::KitModuleAttrDocEmitter(const RecordKeeper &records)
    : KitAttrDocEmitter(records) {}

static TableGen::Emitter::OptClass<KitModuleAttrDocEmitter>
    X("gen-kit-module-attr-doc",
      "Generate documentation for Kitsune-specific module attributes");
