//===- RTNestUtils.h - Utilities to support loop nests ---------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to support parallel execution of tapir loop nests.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_SHARED_RT_NEST_UTILS_H
#define KITSUNE_SHARED_RT_NEST_UTILS_H

// NOTE: There used to be more here, but it got moved around. If we ever add
// something back in, this note should be removed.

#define NEST_MAX_DEPTH 3U

#ifdef __cplusplus

// Since GPU threads can be logically grouped along at most 3 dimensions, the
// maximum depth of a tapir loop nest can be at most 3. Although, in principle,
// there is no such limit on CPU's, the current implementation can divide the
// available CPU threads across at most 3 dimensions. Since changing the maximum
// nest depth will require non-trivial changes to several parts of the
// implementation, we add this sanity check.
static_assert(NEST_MAX_DEPTH == 3 &&
              "Maximum depth of nested tapir loop nests is expected to be 3");

#endif // __cplusplus

#endif // KITSUNE_SHARED_RT_NEST_UTILS_H
