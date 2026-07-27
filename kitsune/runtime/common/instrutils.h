//===- instrutils.h - Base class for runtime instrumentation ----*- C++ -*-===//
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
// Utilities for the instrumentation supported by Kitsune's runtime
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_COMMON_INSTR_UTILS_H
#define KITRT_COMMON_INSTR_UTILS_H

#include "common/env.h"
#include "common/logging.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <vector>

namespace kitrt {

FILE *getInstrOutFile(const char *envVarOutFile);

template <typename KitInstrContext>
void writeEpochs(
    FILE *fp,
    const std::vector<const typename KitInstrContext::EpochImpl *> &epochs) {
  using EpochID = typename KitInstrContext::EpochID;
  using EpochImpl = typename KitInstrContext::EpochImpl;
  using ThreadID = typename KitInstrContext::ThreadID;

  std::optional<EpochID> currEpoch = std::nullopt;
  std::optional<ThreadID> currThrd = std::nullopt;
  bool firstThrd = false;

  fprintf(fp, "{");
  for (const EpochImpl *epoch : epochs) {
    if (currEpoch != epoch->id()) {
      if (currEpoch) {
        fprintf(fp, "\n    ]");
        fprintf(fp, "\n  },");
      }
      fprintf(fp, "\n  \"%s\": {", epoch->name().c_str());
      currEpoch = epoch->id();
      currThrd = std::nullopt;
    }

    if (currThrd != epoch->thrd()) {
      if (currThrd)
        fprintf(fp, "\n    ],");
      fprintf(fp, "\n    \"%ld\": [", epoch->thrd());
      currThrd = epoch->thrd();
      firstThrd = true;
    }

    if (!firstThrd)
      fprintf(fp, ",");

    KitInstrContext::writeEpoch(fp, *epoch);
    firstThrd = false;
  }
  fprintf(fp, "\n    ]");
  fprintf(fp, "\n  }");
  fprintf(fp, "\n}");
  fprintf(fp, "\n");
  fclose(fp);
}

template <typename KitInstrContext>
void writeInstrumentation(const KitInstrContext &ctx) {
  using EpochImpl = typename KitInstrContext::EpochImpl;

  if (ctx.empty())
    return;

  std::vector<const EpochImpl *> epochs;
  for (const EpochImpl &epoch : ctx)
    epochs.push_back(&epoch);

  std::stable_sort(epochs.begin(), epochs.end(),
                   [](const EpochImpl *l, const EpochImpl *r) -> bool {
                     if (l->id() < r->id())
                       return true;
                     else if (l->id() == r->id())
                       return l->thrd() < r->thrd();
                     return false;
                   });

  if (FILE *fp = getInstrOutFile(KitInstrContext::envVarOutFile))
    writeEpochs<KitInstrContext>(fp, epochs);
}

} // namespace kitrt

#endif // KITRT_COMMON_INSTR_UTILS_H
