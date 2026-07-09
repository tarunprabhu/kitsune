//===- kitpapi.h - Kitsune's convenience interface for PAPI ----*- C++ -*--===//
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
// These utilities are a convenience wrapper to allow us to add quick
// instrumentation to record hardware counters using PAPI.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_COMMON_KITPAPI_H
#define KITRT_COMMON_KITPAPI_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/// An opaque structure containing the context for a single PAPI event set.
/// The typical use of this object is as follows:
///
///     PAPIContext *ctx = __kitpapi_new("some-useful-name");
///     __kitpapi_add_event(ctx, <event-name>);
///
///     // Add as many events as needed. Note that individual platforms will
///     // have limits on how many events can be counted simultaneously.
///
///     __kitpapi_start(ctx);
///
///     <code for which hardware counters are desired>
///
///     __kitpapi_stop(ctx);
///
struct KitPAPIContext;
typedef struct KitPAPIContext KitPAPIContext;

// The names of the PAPI events that we support. This is a subset of those
// actually available in PAPI. This interface is intended for Kitsune's
// developers to do some quick instrumentation, so it is unlikely that this list
// will ever be comprehensive.
#define __kitpapi_l1_dcm "l1dcm"     // PAPI_L1_DCM
#define __kitpapi_l2_dcm "l2dcm"     // PAPI_L2_DCM
#define __kitpapi_l3_dcm "l3dcm"     // PAPI_L3_DCM
#define __kitpapi_l1_icm "l1icm"     // PAPI_L1_ICM
#define __kitpapi_l2_icm "l2icm"     // PAPI_L2_ICM
#define __kitpapi_l3_icm "l3icm"     // PAPI_L3_ICM
#define __kitpapi_l1_tcm "l1tcm"     // PAPI_L1_TCM
#define __kitpapi_l2_tcm "l2tcm"     // PAPI_L2_TCM
#define __kitpapi_l3_tcm "l3tcm"     // PAPI_L3_TCM
#define __kitpapi_tot_inst "inst"    // PAPI_TOT_INS
#define __kitpapi_vec_inst "vec"     // PAPI_VEC_INS
#define __kitpapi_load_inst "load"   // PAPI_LD_INS
#define __kitpapi_store_inst "store" // PAPI_SR_INS
#define __kitpapi_br_inst "br"       // PAPI_BR_INS
#define __kitpapi_int_inst "int"     // PAPI_INT_INS
#define __kitpapi_fp_inst "fp"       // PAPI_FP_INS
#define __kitpapi_fma_inst "fma"     // PAPI_FMA_INS
#define __kitpapi_tot_cyc "cyc"      // PAPI_TOT_CYC
#define __kitpapi_ref_cyc "refcyc"   // PAPI_REF_CYC

/// Initialize the PAPI library.
void __kitpapi_initialize(void);

/// Initialize threading support in PAPI. \p getThreadID is a pointer to a
/// function that returns the ID of the thread from which it is called.
void __kitpapi_initialize_threading(void *getThreadID);

/// Clean up the PAPI library.
void __kitpapi_finalize(void);

/// Handle an error returned by a call to a PAPI API function. This will print a
/// warning message to stderr. \p what is an optional label to print before
/// printing the actual PAPI error message. \p err is the error code returned
/// by PAPI.
void __kitpapi_error(const char *what, int err);

/// Create a new PAPI context. \p name is a name for the context. \p name may be
/// nullptr, but providing one is recommended. The optional arguments should
/// each be Kitsune-specific names of events to be recorded. The last optional
/// argument must be nullptr. The example below creates an KitPAPIContext
///
///     KitPAPIContext *ctx = __kitpapi_new("<some-name>", nullptr);
///
/// The example below will create a KitPAPIContext with two events.
///
///     KitPAPIContext *ctx = __kitpapi_new("name", "inst", "cyc", nullptr);
///
/// In each case, additional events can be added to the context with
/// __kitpapi_add_event.
KitPAPIContext *__kitpapi_new(const char *name, ...);

/// Add an event to the event set represented by the context \p ctx. \p evtName
/// is the Kitsune-specific name of the event. If the event could not be added
/// to the event set for any reason, a warning message will be written to stderr
/// and \p ctx will remain unchanged. Note that the behavior of this function,
/// and the subsequent behavior of __kitpapi_start and kitpapi_stop are
/// undefined if \p evtName is an event that is not available on the system
/// where PAPI is used.
void __kitpapi_add_event(KitPAPIContext *ctx, const char *evtName);

/// Start collecting events. \p ctx is a context previously created by a
/// call to __kitpapi_new.
void __kitpapi_start(KitPAPIContext *ctx);

/// Stop collecting events. \p ctx is a context previously created by a call to
/// __kitpapi_new. This will print the counters that were collected to stderr.
/// \p ctx will be destroyed.
void __kitpapi_stop(KitPAPIContext *ctx);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // KITRT_COMMON_KITPAPI_H
