//===- global.cpp - Singleton object for global data ----------------------===//
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
// Singleton object for global data.
//
//===----------------------------------------------------------------------===//

#include "global/global.h"

using namespace kitrt;

KitRTContext::KitRTContext() {
  // We don't do anything here because we don't know when this will run relative
  // to the other global constructors. The *_initialize functions will ensure
  // that any non-pointer fields are initialized correctly. By the time this is
  // used, the fields will have been set correctly.
}

void KitRTContext::setInitialized(bool initialized) {
  this->initialized = initialized;
}

void KitRTContext::setVerbose(bool verbose) { this->verbose = verbose; }

void KitRTContext::setColors(bool colors) { this->colors = colors; }

void KitRTContext::addContext(KitCudaContext *ctx) { this->cuda = ctx; }

void KitRTContext::addContext(KitHipContext *ctx) { this->hip = ctx; }

void KitRTContext::addContext(KitOCilkContext *ctx) { this->ocilk = ctx; }

void KitRTContext::addContext(KitOMPContext *ctx) { this->omp = ctx; }

void KitRTContext::addContext(KitPAPIContext *ctx) { this->papi = ctx; }

void KitRTContext::addContext(KitPthrContext *ctx) { this->pthr = ctx; }

void KitRTContext::addContext(KitQthrContext *ctx) { this->qthr = ctx; }

void KitRTContext::addContext(KitTimerContext *ctx) { this->timer = ctx; }

template <> KitCudaContext *KitRTContext::takeContext() { return take(cuda); }

template <> KitHipContext *KitRTContext::takeContext() { return take(hip); }

template <> KitOCilkContext *KitRTContext::takeContext() { return take(ocilk); }

template <> KitOMPContext *KitRTContext::takeContext() { return take(omp); }

template <> KitPAPIContext *KitRTContext::takeContext() { return take(papi); }

template <> KitPthrContext *KitRTContext::takeContext() { return take(pthr); }

template <> KitQthrContext *KitRTContext::takeContext() { return take(qthr); }

template <> KitTimerContext *KitRTContext::takeContext() { return take(timer); }

namespace kitrt {

/// The global context object.
KitRTContext singleton;

/// The const ref to that object that is exposed.
const KitRTContext &gctx = singleton;

} // namespace kitrt

KitRTContext &kitrt::mutCtx() { return kitrt::singleton; }
