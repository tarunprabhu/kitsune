//===- Instrumentation.h - Utilities to support instrumentation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to support instrumentation of code.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_INSTRUMENTATION_H
#define KITSUNE_CORE_INSTRUMENTATION_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"

namespace llvm {

class raw_ostream;

/// The kinds of instrumentation currently supported in Kitsune. The values of
/// the enum members are flags that can OR'ed into a bitmask.
enum class InstrumentKind : uint32_t {
  Generic = 0x1, ///< Generic instrumentation
  PAPI = 0x2,    ///< Instrumentation to collect hardware counters with PAPI
  Timer = 0x4,   ///< Instrumentation to time Kitsune-specific constructs

  // Sentinels to allow iterating over the members. These should be updated
  // whenever a new member is added.
  SentinelFirst = Generic,
  SentinelLast = Timer,
};

/// To which entities should Kitsune's instrumentation should be added. The
/// values of the enum members are flags that can be OR'ed into a bitmask.
enum class InstrumentUnit : uint32_t {
  /// Instrument the body of tapir loops such that the "start" and "stop"
  /// functions of the instrumentation framework are run on each thread. This
  /// is currently only available on CPU-centric parallel tapir targets.
  /// Clearly, this means that tapir loops with the 'serial' tapir target will
  /// not be instrumented.
  ///
  /// NOTE: The value of this member must *NOT* be changed and must always be
  /// 0x1. Other parts of the code rely on this to work correctly.
  Thread = 0x1,

  /// Instrument only tapir loops.
  Loop = 0x2,

  // Sentinels to allow iterating over the members. These should be updated
  // whenever a new member is added.
  SentinelFirst = Thread,
  SentinelLast = Loop,
};

/// Instrumentation options.
class KitInstrOptions {
private:
  /// The kinds of instrumentation to add. If this is 0, instrumentation should
  /// not be added.
  uint32_t kinds = 0;

  /// The units to which to add instrumentation.
  uint32_t units = 0;

  /// The names of the entities to which to attach instrumentation.
  SmallSet<std::string, 0> names;

  /// The PAPI events to record.
  SmallSet<std::string, 0> papiEvts;

public:
  KitInstrOptions() = default;

  operator bool() const { return enabled(); }

  void addKind(InstrumentKind kind) { kinds |= static_cast<uint32_t>(kind); }

  void addUnit(InstrumentUnit unit) { units |= static_cast<uint32_t>(unit); }

  void addName(StringRef name) { names.insert(name.str()); }

  void addPAPIEvent(StringRef evtName) { papiEvts.insert(evtName.str()); }

  // Override existing units and set all units.
  void setUnitsAll();

  // Override existing units and set only the default units.
  void setUnitsDefault();

  /// Has instrumentation been enabled.
  bool enabled() const { return kinds; }

  /// Has the given instrumentation kind been enabled.
  bool enabled(InstrumentKind kind) const { return kinds & unsigned(kind); }

  /// Has instrumentation for the given unit been enabled.
  bool enabled(InstrumentUnit unit) const { return units & unsigned(unit); }

  /// Should an entity with the given name be instrumented.
  bool shouldInstrument(StringRef name) const;

  /// Get the names of entities to which instrumentation is to be attached.
  iterator_range<decltype(names)::const_iterator> getNames() const {
    return names;
  }

  /// Get the PAPI events to be recorded.
  iterator_range<decltype(papiEvts)::const_iterator> getPAPIEvents() const {
    return papiEvts;
  }

  /// Get the number of PAPI events to be recorded.
  unsigned getNumPAPIEvents() const { return papiEvts.size(); }

  /// Get the kinds of instrumentation that should be added.
  SmallVector<InstrumentKind, 1> getKinds() const;

  /// Get the units to which instrumentation should be added. If not units have
  /// been set explicitly, return the default set of units.
  SmallVector<InstrumentUnit, 1> getUnits() const;

  /// Dump the options using the given output stream. This is useful for
  /// debugging and testing, but not much else.
  void print(raw_ostream &os) const;

public:
  /// Create an instrument options object from command-line options. These are
  /// the LLVM options known to opt. This will always return a valid object, but
  /// the object will have to be checked to determine if instrumentation has
  /// been enabled.
  static KitInstrOptions createFromCommandLine();
};

static_assert(static_cast<uint32_t>(InstrumentKind::SentinelFirst) == 0x1,
              "Integer value of first member of InstrumentKind must be 1");

static_assert(static_cast<uint32_t>(InstrumentUnit::SentinelFirst) == 0x1,
              "Integer value of first member of InstrumentUnit must be 1");
static_assert(static_cast<uint32_t>(InstrumentUnit::Thread) == 0x1 &&
              "Integer value of InstrumentUnit::Thread must always be 1");

} // namespace llvm

#endif // KITSUNE_CORE_INSTRUMENTATION_H
