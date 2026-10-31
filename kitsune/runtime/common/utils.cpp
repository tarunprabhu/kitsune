//===- utlis.cpp - Miscellaneous utilities --------------------------------===//
//
// Copyright (c) 2021, Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021. Los Alamos National Security, LLC. This software was
//  produced under U.S. Government contract DE-AC52-06NA25396 for Los
//  Alamos National Laboratory (LANL), which is operated by Los Alamos
//  National Security, LLC for the U.S. Department of Energy. The
//  U.S. Government has rights to use, reproduce, and distribute this
//  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
//  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
//  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
//  derivative works, such modified software should be clearly marked,
//  so as not to confuse it with the version available from LANL.
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
// Utilities that are used by several tapir-target-specific runtimes.
//
// TODO: Rather than just dumping a number of unrelated utilities here, we
// should try to find a more reasonable organization across source files.
//
//===----------------------------------------------------------------------===//

#include "common/utils.h"
#include "common/env.h"
#include "common/logging.h"
#include "common/unreachable.h"

#include <execinfo.h>
#include <thread>

using namespace kitrt;

uint32_t kitrt::getNumThreadsOrCPUs(const char *alternate) {
  auto warnIfExists = [](const char *envVar) -> void {
    if (envContains(envVar))
      WARN("Ignoring environment variable '%s' with invalid value \"%s\"",
           envVar, envLookup(envVar)->c_str());
  };

  if (std::optional<uint32_t> threads = envLookup<uint32_t>(envNumThreads)) {
    if (*threads > 0) {
      LOG("Environment contains %s=%d", envNumThreads, *threads);
      return *threads;
    }
  }

  // If the environment variable was set, but to an invalid value, issue a
  // diagnostic since the behavior can be confusing if kitrt silently ignores
  // variable and the user does not realize their mistake.
  warnIfExists(envNumThreads);

  if (alternate) {
    if (std::optional<uint32_t> threads = envLookup<uint32_t>(alternate)) {
      // If the value is set to zero, don't use it. This is just to be
      // consistent with the behavior of KIT_NUM_THREADS. It is possible that
      // the underlying runtime can handle the value, but we don't want to take
      // that chance.
      if (*threads > 0) {
        LOG("Environment contains %s=%d", alternate, *threads);
        return *threads;
      }
    }
    warnIfExists(alternate);
  }

  return getNumCPUs();
}

uint32_t kitrt::getNumCPUs(void) {
  // The standard says that std::thread::hardware_concurrency() should only be
  // considered a hint. But it seems to work on the platforms that we care
  // about. Still, it might be worth using a more reliable method.
  if (unsigned cpus = std::thread::hardware_concurrency())
    return cpus;

  WARN("Could not determine number of CPUs. Defaulting to 1");
  return 1;
}

void kitrt::printStackTrace(void) {
  constexpr int depth = 25;
  void *trace[depth];
  int size = backtrace(trace, depth);
  if (char **strings = backtrace_symbols(trace, size)) {
    LOG("stack trace (%d frames)", size);
    for (int i = 0; i < size; ++i)
      LOG("  %s", strings[i]);
    LOG("end stack trace");
    free(strings);
  }
}

uint32_t kitrt::nearestPowerOf2LE(uint32_t n) {
  uint64_t p = 1;
  while (p <= n)
    p <<= 1;
  return p >> 1;
}

uint32_t kitrt::nearestMultipleLT(uint32_t m, uint32_t n) {
  if (m && n && m < n)
    return ((n - 1) / m) * m;
  return m;
}

using Tuple2 = std::tuple<unsigned, unsigned>;
using Tuple3 = std::tuple<unsigned, unsigned, unsigned>;

static void swap(unsigned &a, unsigned &b) {
  unsigned t = a;
  a = b;
  b = t;
}

static Tuple3 sortDescending(unsigned f1, unsigned f2, unsigned f3) {
  if (f1 < f2)
    swap(f1, f2);
  if (f2 < f3)
    swap(f2, f3);
  if (f1 < f3)
    swap(f1, f3);
  return {f1, f2, f3};
}

static Tuple2 sortDescending(unsigned f1, unsigned f2) {
  if (f1 < f2)
    swap(f1, f2);
  return {f1, f2};
}

// Check if the given factors are acceptable. The number that they are factors
// for doesn't matter - but their relative values do.
static bool acceptable(unsigned f1, unsigned f2, bool preferSquare) {
  // We always want to avoid having one of the factors be 1 because we generally
  // want to have more than one thread along every dimension.
  if (f2 == 1)
    return false;

  // When trying to find a square, we allow one of the factors to be at most
  // twice the other. This is a somewhat arbitrary choice - there may be better
  // ones. In reality, we don't expect to have to factorize "unusual" numbers
  // such as primes, odd numbers like 51. or even numbers that don't correspond
  // to the core counts found on most CPU's. Given that, this works well enough.
  if (preferSquare)
    return f1 <= 2 * f2;

  return true;
}

// Find a pair of factors for \p n, where one of the factors is less than or
// equal to \p f. \p f may or may not already be a factor of \p n. The factors
// are returned in descending order. If \p preferSquare is true, we try to
// find factors that are closer in magnitude to one another over simply
// returning the first pair that is found.
static Tuple2 factorize(unsigned n, unsigned f, bool preferSquare) {
  if (f == 1)
    return {n, 1};

  auto [f1, f2] = sortDescending(f, n / f);
  if (acceptable(f1, f2, preferSquare) && (f1 * f2 == n))
    return {f1, f2};
  return factorize(n, f - 1, preferSquare);
}

// Try to factorize \p n as the product of three integers, `f1`, and `f2` where
// `f1 >= f2`. If preferSquare is true, we will try harder to find a pair that
// is closer in magnitude. In general, this will avoid finding factors where
// `f2 == 1`. Given these constraints, the product of the two values returned by
// this function may not equal \p n. For instance, if `n == 33`, if
// \p preferSquare is true, this will return [8,4] which are factors for 32,
// while if \p preferSquare is false, this will return [11,3]. Similarly, for
// `n == 7`, this will always return [3,2]. For n < 4, though, this will return
// the expected factors.
static Tuple2 factor2(unsigned n, bool preferSquare) {
  switch (n) {
  case 1: return {1, 1};
  case 2: return {2, 1};
  case 3: return {3, 1};
  default: break;
  }

  // We could start looking for factors anywhere, but starting with the square
  // root is "reasonable".
  auto [f1, f2] = factorize(n, sqrt(n), preferSquare);

  if (acceptable(f1, f2, preferSquare))
    return {f1, f2};
  return factor2(n - 1, preferSquare);
}

// Factorize \p n in to 3 values where one of the factors is less than or equal
// to \p f.
static Tuple3 factor3(unsigned n, unsigned f, bool preferSquare) {
  if (f == 1)
    return {n, 1, 1};

  // We always want to avoid having one of the factors be 1 because we generally
  // want to have more than one thread along every dimension. After the sort,
  // f1 >= f2 >= f3, so we only need to check that the f3 is 1.
  auto [f2t, f3t] = factor2(n / f, preferSquare);
  auto [f1, f2, f3] = sortDescending(f, f2t, f3t);
  if ((f3 != 1) && (f1 * f2 * f3 == n))
    return {f1, f2, f3};

  return factor3(n, f - 1, preferSquare);
}

// Try to factorize \p n as the product of three integers, `f1`, `f2`, and `f3`
// where `f1 >= f2 >= f3`. If \p preferSquare is true, this will prefer to find
// minimize the difference between the factors. In general, this will avoid
// finding factors where `f3 == 1`. Given these constraints, the product of the
// three values returned by this function may not equal \p n.
static Tuple3 factor3(unsigned n, bool preferSquare) {
  // We special case these because one of the factors here has to be 1,
  // something that the rest of the implementation tries (a bit too hard, as it
  // turns out) to avoid.
  switch (n) {
  case 1: return {1, 1, 1};
  case 2: return {2, 1, 1};
  case 3: return {3, 1, 1};
  case 4: return {2, 2, 1};
  case 5: return {2, 2, 1};
  case 6: return {3, 2, 1};
  case 7: return {3, 2, 1};
  default: break;
  }

  // Start looking at the nearest integer that may be greater than the square
  // root. In some cases, especially when preferSquare is true, this can yield
  // more desirable results than the default of floor(cbrt(n)).
  auto [f1, f2, f3] = factor3(n, ceilf(cbrt(n)), preferSquare);

  // If f1 == n, then \p n is probably prime, or could not be factorized into an
  // "acceptable" triple, (given the constraints that a factor of 1 should be
  // avoided, and the value of preferSquare).
  if (f1 != n)
    return {f1, f2, f3};

  return factor3(n - 1, preferSquare);
}

static LoopNestContext *makeLoopNestContext(uint32_t totalThreads,
                                            uint32_t depth, bool preferSquare) {
  if (depth == 1) {
    return new LoopNestContext(depth, totalThreads);
  } else if (depth == 2) {
    auto [t1, t2] = factor2(totalThreads, preferSquare);
    return new LoopNestContext(depth, t1, t2);
  } else if (depth == 3) {
    auto [t1, t2, t3] = factor3(totalThreads, preferSquare);
    return new LoopNestContext(depth, t1, t2, t3);
  }
  UNREACHABLE("Incorrect loop depth");
}

LoopNestContext *kitrt::nestBegin(uint64_t totalThreads, uint32_t depth) {
  // FIXME: Provide a way to override this preference - perhaps using an
  // environment variable.
  bool preferSquare = true;
  LoopNestContext *ctx = makeLoopNestContext(totalThreads, depth, preferSquare);

  return ctx;
}

void kitrt::nestEnd(LoopNestContext *ctx) { delete ctx; }

uint64_t kitrt::nestThreads(const LoopNestContext &ctx) {
  return ctx.usedThreads;
}

static LoopBounds nestBounds(uint64_t thrds, uint64_t tidx, uint64_t end) {
  const uint64_t ipt = (end + thrds - 1) / thrds;
  const uint64_t b = tidx * ipt;
  const uint64_t e = std::min(b + ipt, end);

  return {b, e};
}

static LoopBounds nestBounds1(const LoopNestContext &ctx, uint64_t tid,
                              uint64_t end) {
  const uint64_t thrds = ctx[0];
  const uint64_t tidx = tid % thrds;

  return nestBounds(thrds, tidx, end);
}

static LoopBounds nestBounds2(const LoopNestContext &ctx, uint64_t tid,
                              uint64_t end, uint32_t level) {
  const uint64_t depth = ctx.depth;
  const uint64_t tn2 = ctx.threads[depth - 1];
  const uint64_t thrds = ctx.threads[level - 1];
  switch (level) {
  case 1: return nestBounds(thrds, tid / tn2, end);
  case 2: return nestBounds(thrds, tid % tn2, end);
  default: break;
  }
  UNREACHABLE("Invalid loop level");
}

static LoopBounds nestBounds3(const LoopNestContext &ctx, uint64_t tid,
                              uint64_t end, uint32_t level) {
  const uint64_t depth = ctx.depth;
  const uint64_t tn3 = ctx[depth - 1];
  const uint64_t tn2 = ctx[depth - 2];
  const uint64_t thrds = ctx[level - 1];
  switch (level) {
  case 1: return nestBounds(thrds, tid / tn3 / tn2, end);
  case 2: return nestBounds(thrds, tid / tn3 % tn2, end);
  case 3: return nestBounds(thrds, tid % tn3, end);
  default: break;
  }
  UNREACHABLE("Invalid loop level");
}

LoopBounds kitrt::nestBounds(const LoopNestContext &ctx, uint64_t tid,
                             uint64_t beg, uint64_t end, uint32_t level) {
  assert(ctx.depth >= 1 && ctx.depth <= NEST_MAX_DEPTH &&
         "Valid loop nest depth");
  assert(level >= 1 && level <= ctx.depth && "Valid loop nest level");

  // The number of threads actually used by the nest may be less than the total
  // number of available threads. But all available threads will still execute
  // the guard of the root of the loop nest. In that case,
  // `tid >= ctx.usedThreads`, but we still need to return bounds. Make sure
  // that what is returned will result in the loop nest being skipped.
  if (tid >= ctx.usedThreads)
    return {0, 0};

  switch (ctx.depth) {
  case 1: return nestBounds1(ctx, tid, end);
  case 2: return nestBounds2(ctx, tid, end, level);
  case 3: return nestBounds3(ctx, tid, end, level);
  default: break;
  }
  UNREACHABLE("Incorrect loop level");
}
