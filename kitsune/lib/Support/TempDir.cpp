//===- TempDir.cpp - RAII object to manage a temporary directory ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RAII object that manages the creation and deletion of a temporary directory.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/TempDir.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;
using namespace llvm::sys;

static void mv(StringRef tempDir, StringRef dest) {
  if (fs::exists(dest)) {
    if (fs::is_directory(dest))
      fs::remove_directories(dest, /*IgnoreErrors=*/true);
    else
      fs::remove(dest, /*IgnoreNonExisting=*/true);
  }
  std::error_code ec = fs::create_directories(dest, /*IgnoreExisting=*/false);
  if (ec) {
    WithColor::error() << "Could not create save directory: " << ec.message()
                       << "\n";
    return;
  }
  fs::rename(tempDir, dest);
  fs::remove_directories(tempDir, /*IgnoreErrors=*/true);
}

TempDir::TempDir(StringRef prefix) {
  fs::createUniquePath(Twine(prefix) + "-%%%%%%%%", tempDir,
                       /*MakeAbsolute=*/true);
  ec = fs::create_directories(tempDir, /*IgnoreExisting=*/false);
  if (ec)
    WithColor::error() << "Could not create temporary directory: "
                       << ec.message() << "\n";
}

TempDir::~TempDir() {
  if (keepAt and keepAt->empty())
    return;

  // If we have to keep this directory, the mv() function can handle
  // non-existing parent directories, and non-empty destinations. However, we do
  // need to compute the correct destination path.
  if (keepAt) {
    SmallString<64> dest(*keepAt);
    if (dest == "$PWD" or dest == "$CWD") {
      fs::current_path(dest);
      path::append(dest, path::filename(tempDir));
    }
    mv(tempDir, dest.str());
  }

  // The temporary directory may have been moved, but the call below will not
  // raise an error.
  fs::remove_directories(tempDir, /*IgnoreErrors=*/true);
}

std::string TempDir::createUniquePath(const Twine &model) const {
  SmallString<64> name, path;
  fs::createUniquePath(model, name, /*MakeAbsolute=*/false);
  path::append(path, tempDir, name);

  return std::string(path);
}
