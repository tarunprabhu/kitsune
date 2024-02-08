//===- KitIntrinsicsDocEmitter.cpp - Emit docs for Kitsune's intrinsics ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generate HTML documentation for Kitsune-specific intrinsics.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "kit-intrinsics-doc"

using namespace llvm;

namespace {

class KitIntrinsicsDocEmitter {
protected:
  // The names of the known documentation categories. This must be kept up to
  // date with any categories that are added. This is the order in which the
  // categories will be added to the generated documentation. This is
  // initialized in the constructor.
  SmallSetVector<StringRef, 4> catNames;

  // The records corresponding to the category definitions. These are used to
  // generate the section headers and intros.
  StringMap<const Record *> categories;

  // The actual intrinsics. These are grouped by category. The intrinsics must
  // be sorted in alphabetical order before use. This is currently done in the
  // constructor.
  SmallDenseMap<const Record *, SmallVector<const Record *, 8>> intrinsics;

  // All records in the file being processed.
  const RecordKeeper &records;

protected:
  StringRef getType(const Record &type) {
    return StringSwitch<StringRef>(type.getName())
        .Case("llvm_void_ty", "void")
        .Case("llvm_i1_ty", "i1")
        .Case("llvm_i8_ty", "i8")
        .Case("llvm_i16_ty", "i16")
        .Case("llvm_i32_ty", "i32")
        .Case("llvm_i64_ty", "i64")
        .Case("llvm_i128_ty", "i128")
        .Case("llvm_half_ty", "f16")
        .Case("llvm_bfloat_ty", "bf16")
        .Case("llvm_float_ty", "float")
        .Case("llvm_double_ty", "double")
        .Case("llvm_f80_ty", "f80")
        .Case("llvm_f128_ty", "f128")
        .Case("llvm_ppcf128_ty", "ppcf128")
        .Case("llvm_ptr_ty", "ptr")
        .Case("llvm_metadata_ty", "metadata")
        .Case("llvm_token_ty", "token")
        .Case("llvm_any_ty", "*")
        .Case("llvm_vararg_ty", "...")
        .Case("llvm_mobile_ptr_ty", "ptr addrspace(67)");
  }

  StringRef getReturnType(const Record &intrinsic) {
    std::vector<const Record *> retTypes =
        intrinsic.getValueAsListOfDefs("RetTypes");
    switch (retTypes.size()) {
    case 0:
      return "void";
    case 1:
      return getType(*retTypes[0]);
    default:
      PrintFatalError(intrinsic.getLoc(),
                      "Kitsune intrinsic cannot return more than one value");
    }
  }

  std::string getLLVMName(const Record &intrinsic) {
    std::string buf;
    raw_string_ostream os(buf);
    StringRef name = intrinsic.getName();

    // The name of the intrinsic will always start with int_.
    os << "llvm.";
    for (char c : name.drop_front(4))
      os << (c == '_' ? '.' : c);

    return buf;
  }

  StringRef getParamName(const Record &paramDoc) {
    return paramDoc.getValueAsString("Name");
  }

  std::string getSignature(const Record &intrinsic) {
    std::string buf;
    raw_string_ostream os(buf);

    os << getReturnType(intrinsic);
    os << " ";
    os << "@" << getLLVMName(intrinsic);

    os << "(";
    std::vector<const Record *> paramTypes =
        intrinsic.getValueAsListOfDefs("ParamTypes");
    if (!paramTypes.empty()) {
      const Record *doc = intrinsic.getValueAsDef("Doc");
      std::vector<const Record *> paramDocs =
          doc->getValueAsListOfDefs("ParamDocs");
      os << getType(*paramTypes[0]) << " %" << getParamName(*paramDocs[0]);
      for (unsigned i = 1; i < paramTypes.size(); ++i) {
        os << ", ";
        os << getType(*paramTypes[i]) << " %" << getParamName(*paramDocs[i]);
      }
    }
    os << ")";
    os << "\n";

    return buf;
  }

  void emitCategory(raw_ostream &os, const Record &cat) {
    StringRef header = cat.getValueAsString("Header");
    std::string border(header.size(), '-');

    os << border << "\n";
    os << header << "\n";
    os << border << "\n";
    os << cat.getValueAsString("Intro") << "\n";
  }

  void emitSignature(raw_ostream &os, const Record &intrinsic) {
    os << ".. code :: kitll\n\n";
    os << "  " << getSignature(intrinsic) << "\n";
  }

  void emitParamDocs(raw_ostream &os, const Record &intrinsic) {
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

  void emitDescr(raw_ostream &os, const Record &intrinsic) {
    const Record *doc = intrinsic.getValueAsDef("Doc");
    os << doc->getValueAsString("Descr") << "\n";
  }

  void emitIntrinsic(raw_ostream &os, const Record &intrinsic) {
    std::string llvmName = getLLVMName(intrinsic);
    std::string border(llvmName.size(), '^');

    os << ".. _" << llvmName << ":\n\n";
    os << border << "\n";
    os << llvmName << "\n";
    os << border << "\n";

    emitSignature(os, intrinsic);
    emitParamDocs(os, intrinsic);
    emitDescr(os, intrinsic);
  }

public:
  KitIntrinsicsDocEmitter(const RecordKeeper &records) : records(records) {
    catNames.insert("KitDocCatMemAlloc");
    catNames.insert("KitDocCatCommon");
    catNames.insert("KitDocCatThreading");
    catNames.insert("KitDocCatGPU");

    for (const Record *r : records.getAllDerivedDefinitions("KitIntrinsic")) {
      const Record *doc = r->getValueAsDef("Doc");
      if (!doc)
        PrintFatalError(r->getLoc(), "Intrinsic must be documented");

      const Record *category = doc->getValueAsDef("Category");
      if (!category)
        PrintFatalError(r->getLoc(), "Missing required category in intrinsic");

      StringRef catName = category->getName();
      if (!catNames.contains(catName))
        PrintFatalError(
            r->getLoc(),
            "INTERNAL ERROR: Category not registered with intrinsic emitter");

      if (!categories.contains(catName)) {
        if (!category->getValueAsOptionalString("Header").has_value())
          PrintFatalError(r->getLoc(), "Category does not have header text");
        if (!category->getValueAsOptionalString("Intro").has_value())
          PrintFatalError(r->getLoc(), "Category does not have intro text");

        categories[catName] = category;
      }

      unsigned numParams = r->getValueAsListOfDefs("ParamTypes").size();
      if (doc->getValueAsListOfDefs("ParamDocs").size() != numParams)
        PrintFatalError(
            doc->getLoc(),
            "Mismatch between number of actual and documented parameters");

      intrinsics[category].push_back(r);
    }

    for (auto &[catName, ints] : intrinsics)
      std::sort(ints.begin(), ints.end(),
                [](const Record *l, const Record *r) -> bool {
                  return l->getName() < r->getName();
                });
  }

  void run(raw_ostream &os) {
    const Record *globalDoc = records.getDef("GlobalDocumentation");
    if (!globalDoc)
      PrintFatalError(
          "The GlobalDocumentation top-level definition is missing, "
          "no documentation will be generated.");

    os << globalDoc->getValueAsString("Intro") << "\n";
    for (StringRef catName : catNames) {
      if (categories.contains(catName)) {
        const Record *category = categories.at(catName);

        emitCategory(os, *category);
        for (const Record *intrinsic : intrinsics.at(category))
          emitIntrinsic(os, *intrinsic);
      }
    }
  }
};

} // namespace

static TableGen::Emitter::OptClass<KitIntrinsicsDocEmitter>
    X("gen-kit-intrinsics-doc",
      "Generate documentation for Kitsune-specific intrinsics");
