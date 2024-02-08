//==- KitCBuiltinsDocEmitter.cpp - Emit docs for Kitsune's C/C++ builtins --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generate HTML documentation for Kitsune-specific C/C++ builtin functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Regex.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kit-cbuiltins-doc"

using namespace llvm;

namespace {

// The prototype of a builtin is expected to be of the form:
//
//   ret(param0, ..., paramN);
//
// We assume that both the return type `ret` and the parameters, `param*` are
// "simple" i.e. they are primitive or pointer types. Pointer types are pointers
// to primitives. Therefore, the primitive names consist exclusively of
// characters in `[A-Za-z0-9_]`. We just parse it with a simple regex.
//
static const Regex re("([^)]+)[(](.*)[)]");

class Prototype {
protected:
  std::string ret;
  SmallVector<std::string, 4> params;

protected:
  static std::string getType(StringRef s) {
    std::string buf;
    raw_string_ostream os(buf);

    if (s.ends_with("!"))
      os << s.drop_back(1) << "* [[kitsune::mobile]]";
    else
      os << s;
    return buf;
  };

public:
  Prototype(const Record &builtin) {
    std::string err;
    SmallVector<StringRef, 4> m;
    if (!re.match(builtin.getValueAsString("Prototype"), &m, &err))
      PrintError(err);

    SmallVector<StringRef, 4> ps;
    SplitString(m[2], ps, ",");

    ret = getType(m[1]);
    for (StringRef p : ps)
      params.push_back(getType(p));
  }

  // Check if the prototype is valid. Since the builtin may not take any
  // arguments, we only check if the return type is a non-empty string. It may
  // not be a "valid" type, but that is not something that we can check here
  // anyway.
  operator bool() const { return ret.size(); }

  StringRef getReturnType() const { return ret; }

  StringRef getParamType(unsigned i) const { return params[i]; }

  unsigned getNumParams() const { return params.size(); }
};

class KitBuiltinsDocEmitter {
protected:
  // The names of the known documentation categories. This must be kept up to
  // date with any categories that are added. This is the order in which the
  // categories will be added to the generated documentation. This is
  // initialized in the constructor.
  SmallSetVector<StringRef, 4> catNames;

  // The records corresponding to the category definitions. These are used to
  // generate the section headers and intros.
  StringMap<const Record *> categories;

  // The actual builtins. These are grouped by category. The builtins must
  // be sorted in alphabetical order before use. This is currently done in the
  // constructor.
  SmallDenseMap<const Record *, SmallVector<const Record *, 8>> builtins;

  // All records in the file being processed.
  const RecordKeeper &records;

protected:
  StringRef getBuiltinName(const Record &builtin) const;
  StringRef getParamName(const Record &paramDoc) const;

  void emitCategory(raw_ostream &os, const Record &cat);
  void emitSignature(raw_ostream &os, const Record &builtin);
  void emitParamDocs(raw_ostream &os, const Record &intrinsic);
  void emitDescr(raw_ostream &os, const Record &builtin);
  void emitBuiltin(raw_ostream &os, const Record &builtin);

public:
  KitBuiltinsDocEmitter(const RecordKeeper &records);
  void run(raw_ostream &os);
};

} // namespace

StringRef KitBuiltinsDocEmitter::getBuiltinName(const Record &builtin) const {
  return builtin.getValueAsListOfStrings("Spellings")[0];
}

StringRef KitBuiltinsDocEmitter::getParamName(const Record &paramDoc) const {
  return paramDoc.getValueAsString("Name");
}

void KitBuiltinsDocEmitter::emitCategory(raw_ostream &os, const Record &cat) {
  StringRef header = cat.getValueAsString("Header");
  std::string border(header.size(), '-');

  os << border << "\n";
  os << header << "\n";
  os << border << "\n";
  os << cat.getValueAsString("Intro") << "\n";
}

void KitBuiltinsDocEmitter::emitSignature(raw_ostream &os,
                                          const Record &builtin) {
  Prototype p(builtin);

  os << ".. code :: c++\n\n";
  os << "  " << p.getReturnType() << " " << getBuiltinName(builtin);
  os << "(";
  if (unsigned numParams = p.getNumParams()) {
    const Record *doc = builtin.getValueAsDef("Doc");
    std::vector<const Record *> paramDocs =
        doc->getValueAsListOfDefs("ParamDocs");
    os << p.getParamType(0) << " " << getParamName(*paramDocs[0]);
    for (unsigned i = 1; i < numParams; ++i) {
      os << ", ";
      os << p.getParamType(i) << " " << getParamName(*paramDocs[i]);
    }
  }
  os << ")";
  os << "\n";
}

void KitBuiltinsDocEmitter::emitParamDocs(raw_ostream &os,
                                          const Record &intrinsic) {
  const Record *doc = intrinsic.getValueAsDef("Doc");
  std::vector<const Record *> params = doc->getValueAsListOfDefs("ParamDocs");
  if (params.size()) {
    os << ".. csv-table::\n";
    os << "  :header: \"\", \"\"\n\n";

    for (const Record *param : params) {
      StringRef name = param->getValueAsString("Name");
      StringRef descr = param->getValueAsString("Descr");

      os << "  " << "\"`" << name << "`\", \"" << descr << "\"\n";
    }
    os << "\n";
  }
}

void KitBuiltinsDocEmitter::emitDescr(raw_ostream &os, const Record &builtin) {
  const Record *doc = builtin.getValueAsDef("Doc");
  os << doc->getValueAsString("Descr") << "\n";
}

void KitBuiltinsDocEmitter::emitBuiltin(raw_ostream &os,
                                        const Record &builtin) {
  StringRef name = getBuiltinName(builtin);
  std::string border(name.size(), '^');

  os << ".. _`" << name << "`:\n\n";
  os << border << "\n";
  os << name << "\n";
  os << border << "\n";

  emitSignature(os, builtin);
  emitParamDocs(os, builtin);
  emitDescr(os, builtin);
}

void KitBuiltinsDocEmitter::run(raw_ostream &os) {
  const Record *globalDoc = records.getDef("GlobalDocumentation");
  if (!globalDoc)
    PrintFatalError("The GlobalDocumentation top-level definition is missing, "
                    "no documentation will be generated.");

  os << globalDoc->getValueAsString("Intro") << "\n";
  for (StringRef catName : catNames) {
    if (categories.contains(catName)) {
      const Record *category = categories.at(catName);

      emitCategory(os, *category);
      for (const Record *builtin : builtins.at(category))
        emitBuiltin(os, *builtin);
    }
  }
}

KitBuiltinsDocEmitter::KitBuiltinsDocEmitter(const RecordKeeper &records)
    : records(records) {
  catNames.insert("KitDocCatMemAlloc");
  catNames.insert("KitDocCatUnsafe");
  catNames.insert("KitDocCatExperimental");

  for (const Record *r : records.getAllDerivedDefinitions("KitCBuiltin")) {
    const Record *doc = r->getValueAsDef("Doc");
    if (!doc)
      PrintFatalError(r->getLoc(), "Builtin must be documented");

    const Record *category = doc->getValueAsDef("Category");
    if (!category)
      PrintFatalError(r->getLoc(), "Missing required category in builtin");

    StringRef catName = category->getName();
    if (!catNames.contains(catName))
      PrintFatalError(
          r->getLoc(),
          "INTERNAL ERROR: Category not registered with builtin emitter");

    if (!categories.contains(catName)) {
      if (!category->getValueAsOptionalString("Header").has_value())
        PrintFatalError(r->getLoc(), "Category does not have header text");
      if (!category->getValueAsOptionalString("Intro").has_value())
        PrintFatalError(r->getLoc(), "Category does not have intro text");

      categories[catName] = category;
    }

    std::vector<StringRef> spellings = r->getValueAsListOfStrings("Spellings");
    if (spellings.size() != 1)
      PrintFatalError(r->getLoc(), "Builtin must have exactly one spelling");

    Prototype p(*r);
    if (!p)
      PrintFatalError(r->getLoc(), "Could not parse builtin prototype");

    unsigned numParams = p.getNumParams();
    if (doc->getValueAsListOfDefs("ParamDocs").size() != numParams)
      PrintFatalError(
          doc->getLoc(),
          "Mismatch between number of actual and documented parameters");

    builtins[category].push_back(r);
  }

  for (auto &[catName, ints] : builtins)
    std::sort(ints.begin(), ints.end(),
              [](const Record *l, const Record *r) -> bool {
                return l->getName() < r->getName();
              });
}

static TableGen::Emitter::OptClass<KitBuiltinsDocEmitter>
    X("gen-kit-cbuiltins-doc",
      "Generate documentation for Kitsune-specific C/C++ builtin functions");
