//===- TempDirTest.cpp - Unit tests for the TempDir helper object ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Support/TempDir.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::sys;

// Check that the temporary directory is created in the constructor and deleted
// deleted in the destructor by default.
TEST(TempDir, basic) {
  std::unique_ptr<TempDir> tempDir;

  tempDir = std::make_unique<TempDir>("kit-basic");
  EXPECT_TRUE(fs::exists(tempDir->getPath()));
  EXPECT_TRUE(fs::is_directory(tempDir->getPath()));

  // The name of the temporary directory must be of the form kit-basic-%%%%%%%%
  // where every % is in [0-9a-f].
  StringRef tempPath = tempDir->getPath();
  StringRef dirname = path::filename(tempPath);

  EXPECT_EQ(dirname.size(), 18U);
  EXPECT_TRUE(dirname.starts_with("kit-basic-"));

  // If the directory was created, there should not be any errors.
  EXPECT_FALSE(tempDir->hasError());
  EXPECT_EQ(tempDir->getError().value(), 0);

  tempDir = nullptr;

  EXPECT_FALSE(fs::exists(tempPath));
}

// Check that createUniquePath returns a usable path within the temporary
// directory.
TEST(TempDir, createUniquePath) {
  std::unique_ptr<TempDir> tempDir = std::make_unique<TempDir>("kit-uniq");
  std::string uniqPath = tempDir->createUniquePath("model-%%%%.o");
  StringRef uniqName = path::filename(uniqPath);

  EXPECT_TRUE(StringRef(uniqPath).starts_with(tempDir->getPath()));
  EXPECT_TRUE(uniqName.starts_with("model-"));
  EXPECT_TRUE(uniqName.ends_with(".o"));
  EXPECT_EQ(uniqName.size(), 12U);

  int fd = -1;
  EXPECT_FALSE(fs::openFileForWrite(uniqPath, fd));

  raw_fd_ostream fs(fd, /*ShouldClose=*/true);
  fs << "Hello";
  fs.close();

  size_t size;
  EXPECT_FALSE(fs::file_size(uniqPath, size));
  EXPECT_EQ(size, 5U);
}

// Check that iterators over the directory contents work as expected.
TEST(TempDir, entries) {
  std::unique_ptr<TempDir> tempDir = std::make_unique<TempDir>("kit-entries");
  std::string f = tempDir->createUniquePath("file-%%%%.txt");
  std::string d = tempDir->createUniquePath("dir-%%%%");

  std::error_code ec;
  raw_fd_ostream fs(f, ec);
  fs << "Hello";
  fs.close();
  fs::create_directories(d, /*IgnoreErrors=*/false);

  EXPECT_EQ(std::distance(tempDir->begin(), tempDir->end()), 2U);
  for (const fs::directory_entry &entry : *tempDir)
    if (entry.type() == fs::file_type::directory_file)
      EXPECT_EQ(entry.path(), d);
    else
      EXPECT_EQ(entry.path(), f);

  for (const fs::directory_entry &entry : tempDir->entries())
    if (entry.type() == fs::file_type::directory_file)
      EXPECT_EQ(entry.path(), d);
    else
      EXPECT_EQ(entry.path(), f);
}

// Calling the keep() method should result in the temporary directory being
// retained even after the object is deleted.
TEST(TempDir, keep) {
  std::unique_ptr<TempDir> tempDir;

  tempDir = std::make_unique<TempDir>("kit-keep");
  EXPECT_TRUE(fs::exists(tempDir->getPath()));
  EXPECT_TRUE(fs::is_directory(tempDir->getPath()));

  StringRef tempPath = tempDir->getPath();
  tempDir->keep();
  tempDir = nullptr;
  EXPECT_TRUE(fs::exists(tempPath));

  fs::remove_directories(tempPath, /*IgnoreErrors=*/true);
  EXPECT_FALSE(fs::exists(tempPath));

  SmallString<64> pwd;
  fs::current_path(pwd);
}

// Calling keep() with $PWD should result in the temporary directory being
// copied into $PWD when the TempDir object goes out of scope.
TEST(TempDir, keepPWD) {
  std::unique_ptr<TempDir> tempDir = std::make_unique<TempDir>("kit-keep-pwd");

  SmallString<64> pwd;
  fs::current_path(pwd);

  StringRef tempPath = tempDir->getPath();
  StringRef tempParent = path::parent_path(tempPath);

  std::string newFile = tempDir->createUniquePath("file-%%%%.txt");
  fs::create_directories(newFile, /*IgnoreErrors=*/false);

  SmallString<64> newDir;
  path::append(newDir, tempParent, "kit-keep-pwd");
  fs::create_directories(newDir, /*IgnoreErrors=*/false);

  SmallString<64> dest;
  path::append(dest, newDir, path::filename(tempPath));

  // This test is run with lit which uses a temporary directory. Set $PWD to
  // a subdirectory of that directory so we can find the moved directory.
  fs::set_current_path(newDir);

  tempDir->keep("$PWD");
  EXPECT_TRUE(fs::exists(tempPath));
  EXPECT_TRUE(fs::exists(newDir));
  EXPECT_FALSE(fs::exists(dest));

  tempDir = nullptr;
  EXPECT_TRUE(fs::exists(newDir));
  EXPECT_TRUE(fs::exists(dest));
  EXPECT_TRUE(fs::is_directory(dest));

  size_t count = 0;
  std::error_code ec;
  fs::directory_iterator it(dest, ec);
  for (fs::directory_iterator end; it != end;) {
    ++count;
    it.increment(ec);
  }
  EXPECT_EQ(count, 1U);

  fs::set_current_path(pwd);
  fs::remove_directories(newDir);
  EXPECT_FALSE(fs::exists(tempPath));
  EXPECT_FALSE(fs::exists(newDir));
  EXPECT_FALSE(fs::exists(dest));
}

// Calling keep() with a directory should result in the temporary directory
// being copied into $PWD when the TempDir object goes out of scope. If the
// save directory does not exist, it should be created. If it exists, it should
// be emptied before the temporary directory is copied into it.
TEST(TempDir, keepAt) {
  StringRef tempPath;
  std::unique_ptr<TempDir> tempDir;

  tempDir = std::make_unique<TempDir>("kit-keep-at");
  tempPath = tempDir->getPath();

  // This test is run with lit which uses a temporary directory. Set $PWD to
  // a subdirectory of that directory so we can find the moved directory.
  StringRef tempParent = path::parent_path(tempDir->getPath());

  std::string newFile = tempDir->createUniquePath("dir-%%%%");
  fs::create_directories(newFile, /*IgnoreErrors=*/false);

  SmallString<64> newDir;
  path::append(newDir, tempParent, "kit-keep-at");

  tempDir->keep(newDir);
  EXPECT_TRUE(fs::exists(tempPath));
  EXPECT_FALSE(fs::exists(newDir));

  tempDir = nullptr;
  EXPECT_FALSE(fs::exists(tempPath));

  {
    size_t count = 0;
    std::error_code ec;
    fs::directory_iterator it(newDir, ec);
    for (fs::directory_iterator end; it != end;) {
      ++count;
      it.increment(ec);
    }
    EXPECT_TRUE(fs::exists(newDir));
    EXPECT_TRUE(fs::is_directory(newDir));
    EXPECT_EQ(count, 1U);
  }

  // Create a new temporary directory. When this goes out of scope, the
  // destination directory should be empty because the temporary directory
  // itself was empty.
  tempDir = std::make_unique<TempDir>("kit-keep-at2");
  tempPath = tempDir->getPath();
  tempDir->keep(newDir);
  tempDir = nullptr;

  {
    size_t count = 0;
    std::error_code ec;
    fs::directory_iterator it(newDir, ec);
    for (fs::directory_iterator end; it != end;) {
      ++count;
      it.increment(ec);
    }
    EXPECT_TRUE(fs::exists(newDir));
    EXPECT_TRUE(fs::is_directory(newDir));
    EXPECT_EQ(count, 0U);
  }

  fs::remove_directories(newDir);
  EXPECT_FALSE(fs::exists(newDir));
}
