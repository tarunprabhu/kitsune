//===- ToString.h - Convert Kitsune-specific types to string ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to convert Kitsune-specific types to string. This could be used to
// convert other types to string as well.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SUPPORT_TO_STRING_H
#define KITSUNE_SUPPORT_TO_STRING_H

#include "kitsune/Core/Instrumentation.h"
#include "kitsune/Core/OptznLevel.h"
#include "kitsune/Core/Tapir.h"
#include "kitsune/Support/MaybeBool.h"
#include "kitsune/Support/TypeTraits.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// \addtogroup kitsune
/// @{

std::string toString(const bool &);
std::string toString(const int8_t &);
std::string toString(const uint8_t &);
std::string toString(const int16_t &);
std::string toString(const uint16_t &);
std::string toString(const int32_t &);
std::string toString(const uint32_t &);
std::string toString(const int64_t &);
std::string toString(const uint64_t &);
std::string toString(const float &);
std::string toString(const double &);
std::string toString(const char *);
std::string toString(const std::string &);
std::string toString(const StringRef &);
std::string toString(const TTID &);
std::string toString(const MaybeBool &);
std::string toString(const OptznLevel &);
std::string toString(const InstrumentKind &);
std::string toString(const InstrumentUnit &);

template <typename T, std::enable_if_t<std::is_iterable_v<T>, int> = 0>
std::string toString(const T &container, StringRef sep = ",") {
  std::string buf;
  raw_string_ostream os(buf);

  bool comma = false;
  for (const auto &v : container) {
    if (comma)
      os << sep;
    os << llvm::toString(v);
    comma = true;
  }
  os.flush();

  return buf;
}

/// Convert the name of the type to a string suitable for printing. For example,
/// int32_t will be rendered to the string "int32_t".
template <typename T> StringRef toString();

/// @}

} // namespace llvm

#endif // KITSUNE_SUPPORT_TO_STRING_H
