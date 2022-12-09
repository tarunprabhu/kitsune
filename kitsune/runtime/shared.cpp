//
//===- gpu_shared.cpp - Kitsune ABI runtime debug support --------------===//
//
// TODO: Need to update LANL/Triad Copyright notice.
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

#include <stdlib.h>
#include <sys/types.h>

#include "kitrt.h"

static unsigned _kitrtDefaultThreadsPerBlock = 256;
static bool _kitrtUseCustomLaunchParameters = false;
static unsigned _kitrtThreadsPerBlock = 0;
static unsigned _kitrtBlocksPerGrid = 0;

void __kitrt_CommonInit() {
  char *envValue;
  if ((envValue = getenv("KITRT_THREADS_PER_BLOCK"))) {
    _kitrtDefaultThreadsPerBlock = atoi(envValue);
    if (__kitrt_verboseMode())
      fprintf(stderr, "kitrt: enviornment threads per block setting = %d.\n",
              _kitrtDefaultThreadsPerBlock);
  }
}

void __kitrt_setDefaultGPUThreadsPerBlock(unsigned threadsPerBlock) {
  _kitrtDefaultThreadsPerBlock = threadsPerBlock;
}

extern "C"
void __kitrt_overrideLaunchParameters(unsigned threadsPerBlock,
                                      unsigned blocksPerGrid) {
  _kitrtUseCustomLaunchParameters = true;
  _kitrtThreadsPerBlock = threadsPerBlock;
  _kitrtBlocksPerGrid   = blocksPerGrid;
}

extern "C"
void __kitrt_resetLaunchParameters() {
  _kitrtUseCustomLaunchParameters = false;
}

void __kitrt_getLaunchParameters(size_t numElements,
                                 unsigned hwWarpSize,
                                 int &threadsPerBlock,
                                 int &blocksPerGrid) {
  if (not _kitrtUseCustomLaunchParameters) {
    unsigned blockSize = 4 * hwWarpSize;
    threadsPerBlock = _kitrtDefaultThreadsPerBlock;
    blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  } else {
    threadsPerBlock = _kitrtThreadsPerBlock;
    blocksPerGrid = _kitrtBlocksPerGrid;
  }
}
