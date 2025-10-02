//===- ObjectUtilsTest.cpp - Tests for Kitsune's object file utilities ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Object/ObjectUtils.h"
#include "CheckUtils.h"
#include "CompressedBinary.h"
#include "kitsune/Object/EmbDeviceCodeContext.h"
#include "llvm/InitializePasses.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

class ObjectUtilsTest : public testing::Test {
protected:
  ObjectUtilsTest() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();

    PassRegistry *Registry = PassRegistry::getPassRegistry();
    initializeCore(*Registry);
    initializeCodeGen(*Registry);
  }
};

TEST_F(ObjectUtilsTest, getNumSections) {
  detail::check_eq(getNumSections(*elfEmpty), 0);
  detail::check_eq(getNumSections(*elfCuda), 3);
  detail::check_eq(getNumSections(*elfHip), 3);
  detail::check_eq(getNumSections(*elfMulti), 4);
  detail::check_eq(getNumSections(*elfSectsSyms), 5);
}

TEST_F(ObjectUtilsTest, getNumSymbols) {
  detail::check_eq(getNumSymbols(*elfEmpty), 0);
  detail::check_eq(getNumSymbols(*elfSectsSyms), 3);
}

TEST_F(ObjectUtilsTest, hasSection) {
  detail::check_false(hasSection(*elfEmpty, ".kit.code.cuda"));
  detail::check_false(hasSection(*elfEmpty, ".kit.code.hip"));
  detail::check_true(hasSection(*elfCuda, ".kit.code.cuda"));
  detail::check_true(hasSection(*elfHip, ".kit.code.hip"));
  detail::check_true(hasSection(*elfMulti, ".kit.code.cuda"));
  detail::check_true(hasSection(*elfMulti, ".kit.code.hip"));
  detail::check_true(hasSection(*elfSectsSyms, ".data"));
  detail::check_true(hasSection(*elfSectsSyms, ".text"));
  detail::check_false(hasSection(*elfSectsSyms, ".comment"));
  detail::check_false(hasSection(*elfSectsSyms, ".bss"));
}

TEST_F(ObjectUtilsTest, hasSymbol) {
  detail::check_false(hasSymbol(*elfEmpty, "x"));
  detail::check_true(hasSymbol(*elfSectsSyms, "x"));
  detail::check_true(hasSymbol(*elfSectsSyms, "get"));
}

TEST_F(ObjectUtilsTest, hasEmbDeviceCode) {
  detail::check_false(hasEmbDeviceCode(*elfEmpty));
  detail::check_true(hasEmbDeviceCode(*elfCuda));
  detail::check_true(hasEmbDeviceCode(*elfHip));
  detail::check_true(hasEmbDeviceCode(*elfMulti));
}

TEST_F(ObjectUtilsTest, getEmbDeviceCodeTTIDs) {
  using Vec = SmallVector<TTID, 0>;

  detail::check_eq(getEmbDeviceCodeTTIDs(*elfEmpty), Vec({}));
  detail::check_eq(getEmbDeviceCodeTTIDs(*elfCuda), Vec({TTID::Cuda}));
  detail::check_eq(getEmbDeviceCodeTTIDs(*elfHip), Vec({TTID::Hip}));

  Expected<SmallVector<TTID, 0>> tts = getEmbDeviceCodeTTIDs(*elfMulti);
  EXPECT_TRUE(bool(tts));

  std::sort(tts->begin(), tts->end());
  EXPECT_EQ(*tts, Vec({TTID::Cuda, TTID::Hip}));
}

TEST_F(ObjectUtilsTest, createEmptyObject) {
  StringRef triple = sys::getDefaultTargetTriple();
  Triple tt(triple);

  if (tt.isOSBinFormatELF()) {
    Expected<OwningBinary<ObjectFile>> objOrErr =
        createEmptyObject(sys::getDefaultTargetTriple(), "empty.o");

    EXPECT_TRUE(bool(objOrErr));

    const ObjectFile &obj = *objOrErr->getBinary();
    for (SectionRef sec : obj.sections()) {
      outs() << "sec: " << *sec.getName() << "\n";
    }

    detail::check_eq(getNumSections(obj), 2);
    detail::check_true(hasSection(obj, ""));
    detail::check_true(hasSection(obj, ".strtab"));
    detail::check_false(hasSection(obj, ".symtab"));
    detail::check_eq(getNumSymbols(obj), 0);
  } else if (tt.isOSBinFormatMachO()) {
    FAIL();
  } else {
    FAIL();
  }
}

TEST_F(ObjectUtilsTest, embedIntoObjectNoSym) {
  StringRef section = ".new.section";
  std::unique_ptr<MemoryBuffer> payload = MemoryBuffer::getMemBuffer("hello");

  Expected<OwningBinary<ObjectFile>> empObjOrErr =
      createEmptyObject(sys::getDefaultTargetTriple());
  EXPECT_TRUE(bool(empObjOrErr));

  Expected<OwningBinary<ObjectFile>> objOrErr =
      embedIntoObject(std::move(*empObjOrErr), std::move(payload), section);
  EXPECT_TRUE(bool(objOrErr));

  const ObjectFile &obj = *objOrErr->getBinary();
  detail::check_eq(getNumSections(obj), 3);
  detail::check_true(hasSection(obj, ""));
  detail::check_true(hasSection(obj, ".strtab"));
  detail::check_false(hasSection(obj, ".symtab"));
  detail::check_true(hasSection(obj, section));

  detail::check_eq(getNumSymbols(obj), 0);
  detail::check_false(hasSymbol(obj, "msg"));
}

TEST_F(ObjectUtilsTest, embedIntoObject2) {
  Expected<OwningBinary<ObjectFile>> empObjOrErr =
      createEmptyObject(sys::getDefaultTargetTriple());
  EXPECT_TRUE(bool(empObjOrErr));

  std::unique_ptr<MemoryBuffer> hello = MemoryBuffer::getMemBuffer("hello");
  Expected<OwningBinary<ObjectFile>> obj1OrErr =
      embedIntoObject(std::move(*empObjOrErr), std::move(hello), ".hello");
  EXPECT_TRUE(bool(obj1OrErr));

  const ObjectFile &obj1 = *obj1OrErr->getBinary();
  detail::check_eq(getNumSections(obj1), 3);
  detail::check_true(hasSection(obj1, ""));
  detail::check_true(hasSection(obj1, ".strtab"));
  detail::check_false(hasSection(obj1, ".symtab"));
  detail::check_true(hasSection(obj1, ".hello"));
  detail::check_false(hasSection(obj1, ".world"));
  detail::check_eq(getNumSymbols(obj1), 0);
  detail::check_false(hasSymbol(obj1, "msg"));

  std::unique_ptr<MemoryBuffer> world = MemoryBuffer::getMemBuffer("world");
  Expected<OwningBinary<ObjectFile>> obj2OrErr =
      embedIntoObject(std::move(*obj1OrErr), std::move(world), ".world", "msg");
  EXPECT_TRUE(bool(obj2OrErr));

  const ObjectFile &obj2 = *obj2OrErr->getBinary();
  detail::check_eq(getNumSections(obj2), 5);
  detail::check_true(hasSection(obj2, ""));
  detail::check_true(hasSection(obj2, ".strtab"));
  detail::check_true(hasSection(obj2, ".symtab"));
  detail::check_true(hasSection(obj2, ".hello"));
  detail::check_true(hasSection(obj2, ".world"));
  detail::check_eq(getNumSymbols(obj2), 1);
  detail::check_true(hasSymbol(obj2, "msg"));
}

TEST_F(ObjectUtilsTest, embedIntoObjectSym) {
  StringRef section = ".new.section";
  std::unique_ptr<MemoryBuffer> payload = MemoryBuffer::getMemBuffer("hello");

  Expected<OwningBinary<ObjectFile>> empObjOrErr =
      createEmptyObject(sys::getDefaultTargetTriple());
  EXPECT_TRUE(bool(empObjOrErr));

  Expected<OwningBinary<ObjectFile>> objOrErr = embedIntoObject(
      std::move(*empObjOrErr), std::move(payload), section, "msg");
  EXPECT_TRUE(bool(objOrErr));

  const ObjectFile &obj = *objOrErr->getBinary();
  detail::check_eq(getNumSections(obj), 4);
  detail::check_true(hasSection(obj, ""));
  detail::check_true(hasSection(obj, ".strtab"));
  detail::check_true(hasSection(obj, ".symtab"));
  detail::check_true(hasSection(obj, section));

  detail::check_eq(getNumSymbols(obj), 1);
  detail::check_true(hasSymbol(obj, "msg"));

  std::error_code ec;
  raw_fd_ostream fs("/tmp/embedded.o", ec);
  fs << obj.getMemoryBufferRef().getBuffer();
  fs.close();
}

// The add* tests actually test EmbDeviceCodeContext::add(ObjectFile). But these
// are tested here since the are defined in kitsune/lib/Object/ObjectUtils.cpp.
TEST_F(ObjectUtilsTest, addEmpty) {
  EmbDeviceCodeContext ctx;
  Expected<unsigned> res = ctx.add(cast<Binary>(*elfEmpty));

  EXPECT_TRUE(bool(res));
  EXPECT_TRUE(*res == 0);
  EXPECT_TRUE(ctx.getTTIDs().empty());
  EXPECT_FALSE(ctx.contains(*elfEmpty));
}

TEST_F(ObjectUtilsTest, addCuda) {
  EmbDeviceCodeContext ctx;
  SmallVector<TTID, 2> tts = {TTID::Cuda};

  Expected<unsigned> res1 = ctx.add(cast<Binary>(*elfCuda));
  EXPECT_TRUE(bool(res1));
  EXPECT_EQ(*res1, 1U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 1UL);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 0UL);
  EXPECT_TRUE(ctx.contains(*elfCuda));

  Expected<unsigned> res2 = ctx.add(cast<Binary>(*elfCuda));
  EXPECT_TRUE(bool(res2));
  EXPECT_EQ(*res2, 0U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 1UL);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 0UL);
}

TEST_F(ObjectUtilsTest, addHip) {
  EmbDeviceCodeContext ctx;
  SmallVector<TTID, 2> tts = {TTID::Hip};

  Expected<unsigned> res1 = ctx.add(cast<Binary>(*elfHip));
  EXPECT_TRUE(bool(res1));
  EXPECT_EQ(*res1, 1U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 0UL);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 1UL);
  EXPECT_TRUE(ctx.contains(*elfHip));

  Expected<unsigned> res2 = ctx.add(cast<Binary>(*elfHip));
  EXPECT_TRUE(bool(res2));
  EXPECT_EQ(*res2, 0U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 0UL);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 1UL);
}

TEST_F(ObjectUtilsTest, addMulti) {
  EmbDeviceCodeContext ctx;
  SmallVector<TTID, 2> tts = {TTID::Cuda, TTID::Hip};

  Expected<unsigned> res1 = ctx.add(cast<Binary>(*elfMulti));
  EXPECT_TRUE(bool(res1));
  EXPECT_EQ(*res1, 2U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 1U);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 1U);
  EXPECT_TRUE(ctx.contains(*elfMulti));

  Expected<unsigned> res2 = ctx.add(cast<Binary>(*elfCuda));
  EXPECT_TRUE(bool(res2));
  EXPECT_EQ(*res2, 1U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 2U);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 1U);
  EXPECT_TRUE(ctx.contains(*elfCuda));

  Expected<unsigned> res3 = ctx.add(cast<Binary>(*elfHip));
  EXPECT_TRUE(bool(res3));
  EXPECT_EQ(*res3, 1U);
  EXPECT_EQ(ctx.getTTIDs(), tts);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 2U);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 2U);
  EXPECT_TRUE(ctx.contains(*elfHip));

  EXPECT_TRUE(ctx.contains(*elfMulti));
}

TEST_F(ObjectUtilsTest, addMemBuf) {
  EmbDeviceCodeContext ctx;
  MemoryBufferRef memBuf = elfHip->getMemoryBufferRef();

  Expected<unsigned> res1 = ctx.add(memBuf);
  EXPECT_TRUE(bool(res1));
  EXPECT_EQ(*res1, 1U);
  EXPECT_EQ(ctx.getTTIDs().size(), 1U);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 1U);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 0U);

  // When adding a memory buffer, we do not check the contents of the buffer, so
  // multiple buffers with identical contents can be added to the context.
  Expected<unsigned> res2 = ctx.add(memBuf);
  EXPECT_TRUE(bool(res2));
  EXPECT_EQ(*res2, 1U);
  EXPECT_EQ(ctx.getTTIDs().size(), 1U);
  EXPECT_EQ(ctx.get(TTID::Hip).size(), 2U);
  EXPECT_EQ(ctx.get(TTID::Cuda).size(), 0U);
}
