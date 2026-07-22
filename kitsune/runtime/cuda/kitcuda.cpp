//===- kitcuda.cpp - Kitsune runtime CUDA support -------------------------===//
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

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <stdbool.h>
#include <sys/syscall.h>

#include "common/env.h"
#include "common/logging.h"
#include "global/global.h"
#include "kitcuda.h"
#include "kitcuda_dylib.h"
#include "kitrt.h"
#include "memory_map.h"

// Global state -- see accessors in kitcuda.h...
bool _kitcuda_initialized = false;
int _kitcuda_device_id = -1;
CUdevice _kitcuda_device = -1;
CUmemLocation _kitcuda_mem_location;
CUcontext _kitcuda_context;

// TODO: We currently don't use these values within the runtime but
// need to do so!
static int _kitcuda_driver_version;
static int _kitcuda_max_threads_per_blk;
static int _kitcuda_warp_size;
static int _kitcuda_supports_gpu_overlap;
static int _kitcuda_supports_concurrent_kerns;
static int _kitcuda_max_regs_per_blk;
static int _kitcuda_major_compute_capability;
static int _kitcuda_minor_compute_capability;

#ifdef KITCUDA_ENABLE_NVTX
const int KIT_NVTX_INIT = 0;
const int KIT_NVTX_MEM = 1;
const int KIT_NVTX_STREAM = 2;
const int KIT_NVTX_LAUNCH = 3;
const int KIT_NVTX_CLEANUP = 4;
#endif // KITCUDA_ENABLE_NVTX

extern "C" {

void __kitcuda_initialize(void) {
  if (_kitcuda_initialized) {
    LOG("Runtime already initialized");
    return;
  }

  // Initialize the shared components of the higher-level runtime.
  __kitrt_initialize();

  LOG("Initializing Kitsune runtime (cuda)");

  KIT_NVTX_PUSH("kitcuda: initialize", KIT_NVTX_INIT);

  if (not __kitcuda_load_symbols()) {
    // TODO: This error block is repetative in the runtime...  Probably best
    // to collapse them down to a call so that we can get consistent messages
    // and mechanisms across the runtime...
    fprintf(stderr, "kitrt: FATAL ERROR - "
                    "unable to resolve dynamic symbols for CUDA!\n");
    fprintf(stderr, "kitrt: aborting.\n");
    kitrt::printStackTrace();
    abort();
  }

  // Standard CUDA initialization steps follow...
  int device_count = 0;
  CU_SAFE_CALL(cuInit_p(0));
  CU_SAFE_CALL(cuDeviceGetCount_p(&device_count));
  if (device_count == 0) {
    fprintf(stderr, "kitcuda: FATAL ERROR - "
                    "no suitable CUDA devices found!\n");
    fprintf(stderr, "kitcuda: aborting.\n");
    kitrt::printStackTrace();
    abort();
  }

  // Note that instead of sharing a common device id across runtime
  // components we instead isolate them within each sub-component;
  // this allows us to think crazy (future) thoughts like running
  // code on both NVIDIA and AMD GPUs.

  // On systems with multiple devices we can select one via the
  // environment.  This can be helpful when chasing issues related
  // to GPU location within a node (e.g. NUMA-ness).
  if (std::optional<int> id = kitrt::envLookup<int>("KITCUDA_DEVICE_ID"))
    _kitcuda_device_id = *id;
  else
    _kitcuda_device_id = 0;

  _kitcuda_mem_location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  _kitcuda_mem_location.id = _kitcuda_device_id;

  assert(_kitcuda_device_id < device_count &&
         "kitcuda: KITCUDA_DEVICE_ID value exceeds available number"
         " of devices.");

  CU_SAFE_CALL(cuDeviceGet_p(&_kitcuda_device, _kitcuda_device_id));
  CU_SAFE_CALL(cuDevicePrimaryCtxRetain_p(&_kitcuda_context, _kitcuda_device));
  CU_SAFE_CALL(cuCtxSetCurrent_p(_kitcuda_context));
  _kitcuda_initialized = true;

  CU_SAFE_CALL(cuDeviceGetAttribute_p(&_kitcuda_max_threads_per_blk,
                                      CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                      _kitcuda_device));
  CU_SAFE_CALL(cuDeviceGetAttribute_p(
      &_kitcuda_warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, _kitcuda_device));
  CU_SAFE_CALL(cuDeviceGetAttribute_p(&_kitcuda_supports_gpu_overlap,
                                      CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
                                      _kitcuda_device));
  CU_SAFE_CALL(cuDeviceGetAttribute_p(&_kitcuda_supports_concurrent_kerns,
                                      CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
                                      _kitcuda_device));
  CU_SAFE_CALL(cuDeviceGetAttribute_p(
      &_kitcuda_max_regs_per_blk, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
      _kitcuda_device));

  CU_SAFE_CALL(cuDriverGetVersion_p(&_kitcuda_driver_version));

  CU_SAFE_CALL(cuDeviceGetAttribute_p(
      &_kitcuda_major_compute_capability,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _kitcuda_device));

  CU_SAFE_CALL(cuDeviceGetAttribute_p(
      &_kitcuda_minor_compute_capability,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _kitcuda_device));

  if (kitrt::gctx.verbose) {
    fprintf(stderr, "    kitcuda: found %d devices.\n", device_count);
    fprintf(stderr, "             using device:     %d\n", _kitcuda_device_id);
    fprintf(stderr, "             driver version:   %d\n",
            _kitcuda_driver_version);
    fprintf(stderr, "             compute capability: %d.%d (sm_%d)\n",
            _kitcuda_major_compute_capability,
            _kitcuda_minor_compute_capability,
            _kitcuda_major_compute_capability * 10 +
                _kitcuda_minor_compute_capability);
    fprintf(stderr, "             warp size:        %d\n", _kitcuda_warp_size);
    fprintf(stderr, "             max threads/blk:  %d\n",
            _kitcuda_max_threads_per_blk);
    fprintf(stderr, "             max regs/blk:     %d\n",
            _kitcuda_max_regs_per_blk);
    fprintf(stderr, "             concurrent kerns: %d\n",
            _kitcuda_supports_concurrent_kerns);
    fprintf(stderr, "             gpu overlap:      %d\n",
            _kitcuda_supports_gpu_overlap);
  }

  // At this point we're ready to go as far as CUDA initialization
  // goes.  The remainder of the initialization checks to see if any
  // environment variables are set that tweak the runtime behavior.

  if (std::optional<int> tpb =
          kitrt::envLookup<int>("KITCUDA_THREADS_PER_BLOCK")) {
    __kitcuda_set_default_threads_per_blk(*tpb);
    if (kitrt::gctx.verbose)
      fprintf(stderr, "  kitcuda: threads/block: %d\n", *tpb);
  }

  if (std::optional<int> tpb =
          kitrt::envLookup<int>("KITCUDA_MAX_THREADS_PER_BLOCK")) {
    __kitcuda_set_max_threads_per_blk(*tpb);
    if (kitrt::gctx.verbose)
      fprintf(stderr, "  kitcuda: max threads/block: %d\n", *tpb);
  }

  if (std::optional<bool> disable_refined_launches =
          kitrt::envLookup<bool>("KITCUDA_DISABLE_LAUNCH_REFINEMENT"))
    if (*disable_refined_launches)
      __kitcuda_enable_launch_refinement(false);

  KIT_NVTX_POP();

  LOG("Initialized Kitsune runtime (cuda)");
}

void __kitcuda_finalize(void) {
  if (not _kitcuda_initialized) {
    LOG("Cannot finalize runtime. Not initialized");
    return;
  }

  LOG("Finalizing Kitsune runtime (cuda)");

  KIT_NVTX_PUSH("kitcuda:finalize", KIT_NVTX_CLEANUP);
  __kitcuda_destroy_thread_streams();
  __kitrt_destroy_memory_map(__kitcuda_mem_destroy);
  // Note that all resources associated with the context will be destroyed.
  CU_SAFE_CALL(cuDevicePrimaryCtxReset_v2_p(_kitcuda_device));
  _kitcuda_initialized = false;
  KIT_NVTX_POP();

  LOG("Finalized Kitsune runtime (cuda)");

  // Finalize the components of Kitsune's runtime that are shared by the
  // tapir-target-specific components.
  __kitrt_finalize();
}

extern "C" uint64_t __kitcuda_num_sms(void) {
  int sms;
  cuDeviceGetAttribute_p(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                         _kitcuda_device);
  return sms;
}

/// The number of partial reductions to perform in parallel.
///
/// \param n The trip count of the parallel loop in containing a reduction
extern "C" uint64_t __kitcuda_reduce_num_partials(uint64_t n) {
  LOG("Calculating number of partial reductions\n");

  // FIXME: This is simply a placeholder to check that the rest of the
  // transformations work as expected. It is beyond terrible for performance, so
  // fix this is ASAP.
  uint64_t numPartials = 8;

  LOG("Number of partial reductions: %ld\n", numPartials);

  return numPartials;
}

// These are declarations from cuda's "internal" headers (internal in the sense
// that they are in headers that are part of the cuda installation, but they
// don't look like they are intended for the average user)
extern "C" void **__cudaRegisterFatBinary(void *fatCubin);
extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle);
extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                  char *deviceAddress, const char *deviceName,
                                  int ext, size_t size, int constant,
                                  int global);
extern "C" void __cudaRegisterManagedVar(void **fatCubinHandle,
                                         void **hostVarPtrAddress,
                                         char *deviceAddress,
                                         const char *deviceName, int ext,
                                         size_t size, int constant, int global);

/// Register a global variable containing device code. We intentionally do not
/// refer to this as a "fat binary" because, at the time of writing, the code is
/// for a single device architecture. This will nearly always be called in the
/// global constructor for Kitsune's runtime.
///
/// \param data Pointer to the global containing the device code.
/// \returns An opaque handle that should be used to register global variables
///          used in the device, and potentially other things.
extern "C" void *__kitcuda_register_devcode(void *data) {
  return __cudaRegisterFatBinary(data);
}

/// Call indicating that all global variables in the device code have been
/// registered. This will nearly always be called in a global constructor for
/// Kitsune's runtime.
/// TODO: Check if calling this is actually necessary. If it is not, remove this
/// function altogether and have Kitsune stop emitting a call to it.
///
/// \param handle An opaque handle returned by __kitcuda_register_devcode.
extern "C" void __kitcuda_register_devcode_end(void **handle) {
  return __cudaRegisterFatBinaryEnd(handle);
}

/// Unregister all device code that was previously registered in a call to
/// __kitcuda_register_devcode.
///
/// \param handle An opaque handle returned by __kitcuda_register_devcode.
extern "C" void __kitcuda_unregister_devcode(void **handle) {
  return __cudaUnregisterFatBinary(handle);
}

/// Register a global variable that is present in the device code that was
/// previously registered with a call to __kitcuda_register_devcode. This will
/// nearly always be called from a global constructor for Kitsune's runtime.
///
/// \param handle An opaque handle returned by __kitcuda_register_devcode.
/// \param hostAddr The address of the global on the host. The device-side
///                 global will be initialized with the value on the host.
/// \param hostName Name of the global variable in host code.
/// \param deviceName Name of the global variable in device code.
/// \param size Size, in the bytes, of the global variable.
/// \param isExternal Is the global variable externally visible.
/// \param isConstant Is the global variable constant.
extern "C" void __kitcuda_register_global(void **handle, void *hostAddr,
                                          char *hostName,
                                          const char *deviceName, size_t size,
                                          int isExternal, int isConstant) {
  // In the declaration for __cudaRegisterVar, the third argument is named
  // "deviceAddr". However, in clang, the name of the global variable is passed.
  // Kitsune does the same thing without any adverse consequences - so far, at
  // least. Maybe there is a definitive source somewhere that will explain what
  // that argument is intended to be, but if it is good enough for clang, it is
  // good enough for us.
  //
  // Per the documentation, the last argument must always be zero.
  return __cudaRegisterVar(handle, (char *)hostAddr, hostName, deviceName,
                           isExternal, size, isConstant, /*global=*/0);
}

/// Register a global variable that is present in the device code that was
/// previously registered with a call to __kitcuda_register_devcode. This will
/// allocate space for the global variable in UVM and return a pointer to that
/// allocated memory via the \p newAddr out variable. This will nearly always
/// be called from a global constructor for Kitsune's runtime.
///
/// NOTE: This function is not currently used.
///
/// \param handle An opaque handle returned by __kitcuda_register_devcode.
/// \param newAddr Out variable that will eventually contain the address, in
///                UVM, for the global variable that will be allocated by this
///                function.
/// \param hostAddr The address of the global in host code.
/// \param deviceName Name of the global variable in the device code.
/// \param size Size, in bytes, of the global variable.
/// \param align The requested alignment of the global variable. This is not
///              currently used, but it is present to keep the signatures of
///              this function and that of the corresponding hip function
///              consistent.
/// \param isExternal Is the global variable externally visible.
/// \param isConstant Is The global variable constant.
extern "C" void
__kitcuda_register_global_managed(void **handle, void **newAddr, void *hostAddr,
                                  const char *deviceName, size_t size,
                                  int align, int isExternal, int isConstant) {
  // In the declaration for __cudaRegisterManagedVar, the third argument is
  // named deviceAddr. But from what we can tell in clang, and from the
  // corresponding function in hip (which generally mirrors Cuda's API), the
  // argument is actually the address of the global variable on the host. Hip's
  // documentation requires it to be the "initial value" for the newly allocated
  // variable, which one would expect to be simply a pointer to the global on
  // the host since it would contain the initial value. The type of the
  // parameter is char*. This is most unhelpful - while it suggests that the
  // variable likely contains a string, it could just as easily be a pointer to
  // arbitrary bytes. For now, we assume that that argument should be a host
  // pointer.
  FATAL("TODO: Check __kitcuda_register_global_managed for correctness");

  // FIXME?: Is it correct for the last argument to be zero? This is the case
  // when registering normal (those that are not allocated in managed memory)
  // globals.
  return __cudaRegisterManagedVar(handle, newAddr, (char *)hostAddr, deviceName,
                                  isExternal, size, isConstant, /*global=*/0);
}

} // extern "C"
