//===- unittests/Driver/ToolChainTest.cpp --- ToolChain tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Kitsune's frontend, interactions with the driver and
// underlying toolchain.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::driver;

namespace {

// Check that the KitsuneFrontend flag is correctly set in the driver.
TEST(KitsuneFrontendTest, KitsuneFrontend) {
  IntrusiveRefCntPtr<DiagnosticOptions> diagOpts = new DiagnosticOptions();
  IntrusiveRefCntPtr<DiagnosticIDs> diagID(new DiagnosticIDs());
  struct TestDiagnosticConsumer : public DiagnosticConsumer {};
  DiagnosticsEngine diags(diagID, &*diagOpts, new TestDiagnosticConsumer);
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs(
      new llvm::vfs::InMemoryFileSystem);

  const char *kitcc = "/home/test/bin/" KITSUNE_C_FRONTEND;
  const char *kitxx = "/home/test/bin/" KITSUNE_CXX_FRONTEND;
  const char *kitfc = "/home/test/bin/" KITSUNE_Fortran_FRONTEND;
  StringRef triple = "arm-linux-gnueabi";

  Driver kcc(kitcc, triple, diags, "Kitsune C compiler", fs);
  EXPECT_TRUE(kcc.BuildCompilation({kitcc, "foo.c"}));
  EXPECT_TRUE(kcc.CCCIsCC());
  EXPECT_TRUE(kcc.IsKitsuneFrontend());

  Driver kxx(kitxx, triple, diags, "Kitsune C++ compiler", fs);
  EXPECT_TRUE(kxx.BuildCompilation({kitxx, "foo.cpp"}));
  EXPECT_TRUE(kxx.CCCIsCXX());
  EXPECT_TRUE(kxx.IsKitsuneFrontend());

  // We can check this even if Fortran support has not been enabled.
  Driver kfc(kitfc, triple, diags, "Kitsune Fortran compiler", fs);
  EXPECT_TRUE(kfc.BuildCompilation({kitfc, "foo.f90"}));
  EXPECT_TRUE(kfc.IsFlangMode());
  EXPECT_TRUE(kfc.IsKitsuneFrontend());

  const char *clang = "/home/test/bin/clang";
  Driver ccc(clang, triple, diags, "clang", fs);
  EXPECT_TRUE(ccc.BuildCompilation({clang, "foo.c"}));
  EXPECT_FALSE(ccc.IsKitsuneFrontend());

  const char *clangxx = "/home/test/bin/clang++";
  Driver cxx(clangxx, triple, diags, "clang++", fs);
  EXPECT_TRUE(cxx.BuildCompilation({clangxx, "foo.cpp"}));
  EXPECT_FALSE(cxx.IsKitsuneFrontend());

  const char *flang = "/home/test/bin/flang";
  Driver ffc(flang, triple, diags, "flang", fs);
  EXPECT_TRUE(ffc.BuildCompilation({flang, "foo.f90"}));
  EXPECT_FALSE(ffc.IsKitsuneFrontend());
}

// Check that the isKitsuneFrontend() returns the correct value depending on
// the frontend used. This checks for the flag in the LangOptions object which
// is different from the driver and must be set correctly as well.w
TEST(KitsuneFrontendTest, KitsuneLangOptions) {
  IntrusiveRefCntPtr<DiagnosticOptions> diagOpts = new DiagnosticOptions();
  IntrusiveRefCntPtr<DiagnosticIDs> diagID(new DiagnosticIDs());
  struct TestDiagnosticConsumer : public DiagnosticConsumer {};
  DiagnosticsEngine diags(diagID, &*diagOpts, new TestDiagnosticConsumer);
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs(
      new llvm::vfs::InMemoryFileSystem);
  CompilerInvocation invoc;
  const char * args[] = {"-cc1", "foo.c"};

  CompilerInvocation::CreateFromArgs(invoc, args, diags, KITSUNE_C_FRONTEND);
  EXPECT_TRUE(invoc.getKitsuneOpts().isKitsuneFrontend());

  CompilerInvocation::CreateFromArgs(invoc, args, diags, KITSUNE_CXX_FRONTEND);
  EXPECT_TRUE(invoc.getKitsuneOpts().isKitsuneFrontend());

  CompilerInvocation::CreateFromArgs(invoc, args, diags, "/bin/clang");
  EXPECT_FALSE(invoc.getKitsuneOpts().isKitsuneFrontend());

  CompilerInvocation::CreateFromArgs(invoc, args, diags, "/bin/clang++");
  EXPECT_FALSE(invoc.getKitsuneOpts().isKitsuneFrontend());

#if KITSUNE_Fortran_ENABLED
  const char* fc1Args[] = {"-fc1", "foo.f90"};

  CompilerInvocation::CreateFromArgs(invoc, fc1Args, diags,
                                     KITSUNE_Fortran_FRONTEND);
  EXPECT_TRUE(invoc.getKitsuneOpts().isKitsuneFrontend());

  CompilerInvocation::CreateFromArgs(invoc, fc1Args, diags, "/bin/flang");
  EXPECT_FALSE(invoc.getKitsuneOpts().isKitsuneFrontend());
#endif // KITSUNE_Fortran_ENABLED
}

} // end anonymous namespace.
