//===- Instrumentation.cpp - Utilities to support instrumentation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to support instrumentation of code. The utilities here can be used
// for both printf-style debugging of passes, and broader instrumentation of
// code.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/Instrumentation.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::list<std::string>
    clKitInstr("kit-instr",
               cl::desc("The list of kinds of instrumentation to add"),
               cl::cat(cl::catKitClOpts), cl::CommaSeparated);

static cl::list<std::string>
    clKitInstrOnly("kit-instr-only",
                   cl::desc("The list of names of the entities to instrument"),
                   cl::cat(cl::catKitClOpts), cl::CommaSeparated);

static cl::list<std::string>
    clKitInstrUnit("kit-instr-unit",
                   cl::desc("The list of units to instrument"),
                   cl::cat(cl::catKitClOpts), cl::CommaSeparated);

static cl::list<std::string>
    clKitInstrPAPI("kit-instr-papi",
                   cl::desc("The list of PAPI events to record"),
                   cl::cat(cl::catKitClOpts), cl::CommaSeparated);

template <> std::string llvm::toString(const InstrumentKind &kind) {
  switch (kind) {
  case InstrumentKind::Generic: return "generic";
  case InstrumentKind::PAPI: return "papi";
  case InstrumentKind::Timer: return "timer";
  }
  llvm_unreachable("toString: InstrumentKind not handled");
}

template <> std::optional<InstrumentKind> llvm::fromString(StringRef s) {
  return StringSwitch<std::optional<InstrumentKind>>(s)
      .Case("generic", InstrumentKind::Generic)
      .Case("papi", InstrumentKind::PAPI)
      .Case("timer", InstrumentKind::Timer)
      .Default(std::nullopt);
}

template <> std::string llvm::toString(const InstrumentUnit &unit) {
  switch (unit) {
  case InstrumentUnit::Thread: return "thread";
  case InstrumentUnit::Loop: return "loop";
  }
  llvm_unreachable("toString: InstrumentUnit not handled");
}

template <> std::optional<InstrumentUnit> llvm::fromString(StringRef s) {
  return StringSwitch<std::optional<InstrumentUnit>>(s)
      .Case("loop", InstrumentUnit::Loop)
      .Case("thread", InstrumentUnit::Thread)
      .Default(std::nullopt);
}

static constexpr uint32_t unitsAllMask = ~0;
static constexpr uint32_t unitsDefaultMask = ~1;

template <typename E> static SmallVector<E, 1> parseFlags(uint32_t flags) {
  SmallVector<E, 1> ret;
  uint32_t first = static_cast<uint32_t>(E::SentinelFirst);
  uint32_t last = static_cast<uint32_t>(E::SentinelLast);
  for (uint32_t flag = first; flags && flag && flag <= last;
       flag <<= 1, flags >>= 1)
    if (flags & 0x1)
      ret.push_back(static_cast<E>(flag));
  return ret;
}

void KitInstrOptions::setUnitsDefault() {
  // By default, all Kitsune-specific constructs are instrumented, but
  // per-thread instrumentation is disabled. We are guaranteed that the value of
  // InstrumentUnit::Thread is 0x1. This will set everything except the
  // right-most bit to 1. It doesn't matter if the higher-order bits are 1, they
  // are ignored for all other purposes.
  units = unitsDefaultMask;
}

void KitInstrOptions::setUnitsAll() {
  // This will set all the bits to 1. It doesn't matter if the higher order bits
  // are 1. They are ignored for all other purposes anyway.
  units = unitsAllMask;
}

bool KitInstrOptions::shouldInstrument(StringRef name) const {
  if (names.empty())
    return true;
  return names.contains(name.data());
}

SmallVector<InstrumentKind, 1> KitInstrOptions::getKinds() const {
  return parseFlags<InstrumentKind>(kinds);
}

SmallVector<InstrumentUnit, 1> KitInstrOptions::getUnits() const {
  return parseFlags<InstrumentUnit>(units ? units : unitsDefaultMask);
}

void KitInstrOptions::print(raw_ostream &os) const {
  os << "Kitsune instrumentation options:\n";
  if (enabled()) {
    os << "  Kinds: " << toString(getKinds()) << "\n";
    os << "  Units: " << toString(getUnits()) << "\n";
    os << "  Only:  " << toString(names) << "\n";
    os << "  PAPI:  " << toString(papiEvts) << "\n";
  } else {
    os << "  Kinds:\n";
    os << "  Units:\n";
    os << "  Only:\n";
    os << "  PAPI:\n";
  }
}

KitInstrOptions KitInstrOptions::createFromCommandLine() {
  KitInstrOptions opts;
  if (!clKitInstr.getNumOccurrences())
    return opts;

  for (StringRef kind : clKitInstr)
    opts.addKind(*llvm::fromString<InstrumentKind>(kind));

  if (clKitInstrUnit.size() == 1 && clKitInstrUnit[0] == "all")
    opts.setUnitsAll();
  else if (clKitInstrUnit.size() == 1 && clKitInstrUnit[0] == "default")
    opts.setUnitsDefault();
  else
    for (StringRef unit : clKitInstrUnit)
      opts.addUnit(*llvm::fromString<InstrumentUnit>(unit));

  for (StringRef name : clKitInstrOnly)
    opts.addName(name);

  for (StringRef evt : clKitInstrPAPI)
    opts.addPAPIEvent(evt);

  return opts;
}
