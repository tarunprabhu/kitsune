//===- KitsuneOptions.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements initialization of KitsuneOptions from the -cc1/-fc1
// command line.
//
//===----------------------------------------------------------------------===//

// NOTE: This is currently in clang/lib/Basic because it is shared between
// clang and flang. But flang would like to remove the dependence on clangBasic.
// The main reason this cannot be moved to llvm/lib/Frontend/Driver to live
// alongside KitsuneOptions.h is the DiagnosticsEngine object which is defined
// in clangBasic. If the LLVM folks upstream move the DiagnosticsEngine where
// it can be shared without needing to resort to clangBasic, we should be able
// to move this file as well.

#include "llvm/Frontend/Driver/KitsuneOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"

#include <sstream>
#include <vector>

using namespace clang;
using namespace clang::driver::options;
using namespace llvm::opt;

using KitsuneOptions = llvm::driver::KitsuneOptions;

