//===- TempDir.h - RAII object to manage a temporary directory --*- C++ -*-===//
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

#ifndef KITSUNE_SUPPORT_TEMP_DIR_H
#define KITSUNE_SUPPORT_TEMP_DIR_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <iterator>
#include <optional>

namespace llvm {

/// RAII object to manage a temporary directory. The temporary directory will be
/// created within the temporary directory of the system. The directory will be
/// deleted when the object goes out of scope, unless a flag to keep the
/// directory has been set. Additional functionality is described in the
/// documentation for the methods provided by this class.
class TempDir {
public:
  /// Iterate over the entries in the temporary directory. This will not recurse
  /// into subdirectories.
  struct entry_iterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using difference_type = unsigned;
    using value_type = sys::fs::directory_entry;
    using pointer = sys::fs::directory_entry *;
    using reference = sys::fs::directory_entry &;

  private:
    sys::fs::directory_iterator it;
    std::error_code &ec;

  public:
    entry_iterator(StringRef path, std::error_code &ec)
        : it(path, ec), ec(ec) {}
    entry_iterator(std::error_code &ec) : ec(ec) {}

    entry_iterator &operator++() {
      it.increment(ec);
      return *this;
    }

    entry_iterator operator++(int) {
      entry_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    const value_type &operator*() const { return *it; }
    const value_type *operator->() const { return &*it; }
    bool operator==(const entry_iterator &o) const { return it == o.it; }
    bool operator!=(const entry_iterator &o) const { return it != o.it; }
  };

private:
  /// Absolute path to the temporary directory.
  SmallString<64> tempDir;

  /// If this is std::nullopt, the temporary directory will be deleted when this
  /// object goes out of scope. Otherwise, several options are available:
  ///
  ///   - If it is set to an empty string, the temporary directory will not be
  ///     deleted.
  ///
  ///   - If it is set to "$PWD" or "$CWD", the temporary directory will be
  ///     moved to the current working directory.
  ///
  ///   - If it is set to a non-empty string, the string is interpreted as the
  ///     path to a directory, D. If the directory does not exist, it will be
  ///     created along with any other directories that need to be created in
  ///     the path. The contents of the temporary directory will be moved into
  ///     this directory and the temporary directory will be deleted. If an
  ///     error occurs at any time in this process - for instance because a
  ///     directory could not be created or a file could not be moved - the
  ///     state of the file system is indeterminate. The temporary directory
  ///     will be deleted, but one or more directories may have been created
  ///     on the path to D.
  ///
  std::optional<std::string> keepAt = std::nullopt;

  /// The error for the last operation that was carried out, if any.
  std::error_code ec;

public:
  /// Create an instance of this class. The temporary directory will be created
  /// immediately. If the directory could not be created, an error message will
  /// be printed to stderr. It is an error to call any method on the object in
  /// this case. The name of the created temporary directory is guaranteed to
  /// start with \ref prefix.
  TempDir(StringRef prefix);
  ~TempDir();

  TempDir() = delete;
  TempDir(const TempDir &) = delete;
  TempDir(TempDir &&) = delete;
  TempDir &operator=(const TempDir &) = delete;
  TempDir &operator=(TempDir &&) = delete;

  /// Check if an error has occurred for any reason. This usually occurs when
  /// creating, or iterating over, the temporary directory.
  bool hasError() const { return bool(ec); }

  /// Get the current error, if any.
  const std::error_code getError() const { return ec; }

  /// Get the path to the temporary directory. This should only be called after
  /// the temporary directory has been created.
  StringRef getPath() const { return tempDir; }

  /// Construct a path by appending one or more elements to that of the
  /// temporary directory. The newly created path is returned.
  template <typename... Args> std::string append(Args &&...args) const {
    SmallString<64> buf;
    sys::path::append(buf, tempDir, args...);
    return std::string(buf);
  }

  /// Get a unique file name for use within the temporary directory. The
  /// generated name is only guaranteed to be unique within the temporary
  /// directory itself. There is currently no way to obtain a unique name for a
  /// file within any subdirectories. The name is based on \ref model with every
  /// "%" replaced by a random character in [0-9a-f]. Callers may create a file
  /// or a directory with the returned path.
  std::string createUniquePath(const Twine &model) const;

  /// Do not delete the temporary directory when this object goes out of scope.
  void keep() { keepAt = ""; }

  /// When this object goes out of scope, move the contents of the temporary
  /// directory to the directory given by \ref dest. If \ref dest does not
  /// exist, it will be created, along with any parent directories that do not
  /// exist. The temporary directory will be deleted after the contents have
  /// been moved. If \ref path is an empty string, the temporary directory, and
  /// its contents, will not be deleted when the object goes out of scope.
  void keep(StringRef dest) { keepAt = std::string(dest); }

  /// Methods to iterate over the contents of the temporary directory.
  /// @{
  entry_iterator begin() { return entry_iterator(tempDir, ec); }
  entry_iterator end() { return entry_iterator(ec); }
  iterator_range<entry_iterator> entries() {
    return iterator_range(begin(), end());
  }
  /// @}
};

} // namespace llvm

template <> struct std::iterator_traits<llvm::TempDir::entry_iterator> {
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = llvm::sys::fs::directory_entry;
  using reference = llvm::sys::fs::directory_entry &;
  using pointer = llvm::sys::fs::directory_entry *;
};

#endif // KITSUNE_SUPPORT_TEMP_DIR_H
