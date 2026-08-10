//===- instrutils.cpp - Utilities for Kitsune's instrumentation support ---===//
//
// Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021, 2023. Los Alamos National Security, LLC. This
//  software was produced under U.S. Government contract
//  DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which
//  is operated by Los Alamos National Security, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT
//  NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
//  OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
//  If software is modified to produce derivative works, such modified
//  software should be clearly marked, so as not to confuse it with
//  the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
//  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
//
// Utilities for compiler-inserted instrumentation backed by Kitsune's runtime.
//
//===----------------------------------------------------------------------===//

#include "common/instr.h"
#include "common/env.h"
#include "common/logging.h"

#include <algorithm>
#include <cstring>

using namespace kitrt;

static void writeJSONHeader(FILE *fp) { fprintf(fp, "{"); }

template <typename Epoch>
static void writeEpochHeader(FILE *fp, const Epoch &epoch) {
  fprintf(fp, "\n  \"%s\": {", epoch.name());
}

template <typename Epoch>
static void writeThreadHeader(FILE *fp, const Epoch &epoch) {
  fprintf(fp, "\n    \"%ld\": [", epoch.thrd());
}

template <typename Epoch> static void writeEpoch(FILE *fp, const Epoch &epoch) {
  epoch.writeJSON(fp);
}

template <typename Epoch>
static void writeThreadFooter(FILE *fp, const Epoch &epoch, bool comma) {
  fprintf(fp, "\n    ]");
  if (comma)
    fprintf(fp, ",");
}

template <typename Epoch>
static void writeEpochFooter(FILE *fp, const Epoch &epoch, bool comma) {
  fprintf(fp, "\n  }");
  if (comma)
    fprintf(fp, ",");
}

template <typename Epoch>
static void writeEpochs(FILE *fp, const std::vector<const Epoch *> &epochs) {
  assert(!epochs.empty() && "Epochs must not be empty");

  writeEpochHeader(fp, *epochs[0]);
  writeThreadHeader(fp, *epochs[0]);
  writeEpoch(fp, *epochs[0]);
  for (unsigned i = 1, e = epochs.size(); i != e; ++i) {
    const Epoch &prev = *epochs[i - 1];
    const Epoch &curr = *epochs[i];
    if (strcmp(prev.name(), curr.name())) {
      writeThreadFooter(fp, prev, /*comma=*/false);
      writeEpochFooter(fp, prev, /*comma=*/true);
      writeEpochHeader(fp, curr);
      writeThreadHeader(fp, curr);
    } else if (prev.thrd() != curr.thrd()) {
      writeThreadFooter(fp, prev, /*comma=*/true);
      writeThreadHeader(fp, curr);
    } else {
      // This is a different iteration on the same thread.
      fprintf(fp, ",");
    }
    writeEpoch(fp, curr);
  }
  writeThreadFooter(fp, *epochs[0], /*comma=*/false);
  writeEpochFooter(fp, *epochs[0], /*comma=*/false);
}

static void writeJSONFooter(FILE *fp) { fprintf(fp, "\n}"); }

template <typename Epoch>
static void writeJSON(FILE *fp, const std::vector<const Epoch *> &epochs) {
  writeJSONHeader(fp);
  writeEpochs(fp, epochs);
  writeJSONFooter(fp);
  fprintf(fp, "\n");
}

// If the environment variable, \p envVar is set to a non-empty string, it is
// assumed to the name (or absolute path) of a file to which the recorded
// instrumentation is to be written. A special case is if the environment
// variable is set to "-". In this case, the FILE object corresponding to
// stdout will be returned. Otherwise, an attempt will be made to open the file
// named by \p envVar for writing. If it succeeds, the contents of the file
// will be deleted and a FILE object pointing to the start of the file will be
// returned. If the file could not be opened for writing for any reason, NULL
// will be returned. if \p envVarOutFile is not set in the environment, a FILE
// object corresponding to stderr will be returned.
static FILE *getOutputFile(const char *envVar) {
  if (std::optional<std::string> fname = envLookup(envVar)) {
    if (fname == "-")
      return stdout;

    LOG("Writing timings to file: %s", fname->c_str());
    FILE *fp = fopen(fname->c_str(), "wt");
    if (!fp)
      WARN("Could not open file for writing");
    return fp;
  } else {
    return stderr;
  }
}

template <typename T, typename EpochT>
KitInstrBase<T, EpochT>::KitInstrBase()
    : separate(envContains("KIT_INSTR_SEPARATE")) {}

template <typename T, typename EpochT>
void KitInstrBase<T, EpochT>::writeJSON(const char *outFileEnvVar) const {
  auto sortByNameThenThreadID = [](const Epoch *l, const Epoch *r) -> bool {
    int cmp = strcmp(l->name(), r->name());
    if (cmp < 0)
      return true;
    else if (cmp == 0)
      return l->thrd() < r->thrd();
    return false;
  };

  if (epochs.empty())
    return;

  // Collect epochs.
  std::vector<const Epoch *> epochs;
  for (const auto &[_, ownedEpochs] : this->epochs)
    for (const std::unique_ptr<Epoch> &epoch : ownedEpochs)
      epochs.push_back(epoch.get());

  // Sort the epochs by name, and then by thread id.
  // TODO: What we actually want here is to first group by name in the order
  // in which the epochs were added, then sort by thread id *only*. This will
  // ensure that the epochs appear in the output in the order in which they
  // were created during execution. This is, arguably, more useful than
  // sorting lexicographically by name. But the complexity of doing that is
  // probably not worth it since the JSON output is not really intended for
  // human consumption anyway.
  std::stable_sort(epochs.begin(), epochs.end(), sortByNameThenThreadID);

  if (FILE *fp = getOutputFile(outFileEnvVar)) {
    ::writeJSON(fp, epochs);
    if (fp != stdout && fp != stderr)
      fclose(fp);
  }
}

// -----------------------------------------------------------------------------
// Instantiate all supported instrumentation runtimes.

#include "papi/kitpapi.h"
#include "timer/timer.h"

// If a new instrumentation runtime is added, it should be instantiated here.
template class kitrt::KitInstrBase<KitPAPIContext, KitPAPIEpoch>;
template class kitrt::KitInstrBase<KitTimerContext, KitTimerEpoch>;
