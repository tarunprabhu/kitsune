//===- kithip.cpp - Kitsune runtime HIP support ---------------------------===//
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

// TODO: Add support for roctracer (see https://github.com/ROCm/roctracer)

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include "kithip.h"
#include "kithip_rtinfo.h"
#include "kitrt.h"
#include "memory_map.h"

namespace kithip_rt {
kithip_rt_info_t rt_info;
}

#ifdef KITHIP_ENABLE_ROCTX
#error "rocTX is not yet supported."
#endif

// All "compiler-facing" calls are C calling convention to avoid having to
// codegen C++ managled names...

extern "C" void __kithip_dump_dev_properties(hipDeviceProp_t &props) {
  using namespace std;
  using namespace kithip_rt;
  cerr << "kitrt[hip]: ###### DEVICE PROPERTIES ######\n";
  cerr << "  GPU name/model: " << props.name << endl;
  cerr << "    - PCI bus info ***\n";
  cerr << "        bus id: " << props.pciBusID << endl;
  cerr << "        device id: " << props.pciDeviceID << endl;
  cerr << "        domain id: " << props.pciDomainID << endl;
  cerr << "   - Compute Unit (CU) info ***\n";
  cerr << "        multi-processor (mp) count: " << props.multiProcessorCount
       << endl;
  cerr << "      max threads / mp: " << props.maxThreadsPerMultiProcessor
       << endl;
  cerr << "      max threads / block: " << props.maxThreadsPerBlock << endl;
  cerr << "      max registers / block: " << props.regsPerBlock << endl;
  cerr << "      warp size: " << props.warpSize << endl;
  cerr << "      max threads x dim: " << props.maxThreadsDim[0] << endl;
  cerr << "      max threads y dim: " << props.maxThreadsDim[1] << endl;
  cerr << "      max threads z dim: " << props.maxThreadsDim[2] << endl;
  cerr << "      max grid x size: " << props.maxGridSize[0] << endl;
  cerr << "      max grid y size: " << props.maxGridSize[1] << endl;
  cerr << "      max grid z size: " << props.maxGridSize[2] << endl;
  cerr << "   *** General memory specs ***\n";
  cerr << "      total constant memory: " << props.totalConstMem << endl;
  cerr << "      shared memory / block: " << props.sharedMemPerBlock << endl;
  cerr << "      L2 cache size: " << props.l2CacheSize << endl;
  cerr << "      total global memory: " << bytesToGBytes(props.totalGlobalMem)
       << " GB" << endl;
}

extern "C" bool __kithip_is_initialized(void) {
  using namespace kithip_rt;
  return rt_info.initialized;
}

extern "C" void __kithip_initialize(void) {
  using namespace kithip_rt;
  if (isInitialized()) {
    LOG("Runtime already initialized");
    return;
  }

  // Initialize the components of kitsune's runtime that are shared by the
  // tapir-target-specific components.
  __kitrt_initialize();

  LOG("Initializing Kitsune runtime (hip)");

  // AMD's documentation suggests that there is no need to explicitly call
  // hipInit() as all API entry points will initialize when necessary.  For now,
  // we'll just follow the more direct path.
  HIP_SAFE_CALL(hipInit(0));

  HIP_SAFE_CALL(hipGetDeviceCount(&rt_info.device_count));
  if (rt_info.device_count <= 0) {
    fprintf(stderr, "kitrt[hip]: no suitable hip-enabled devices found!\n");
    kitrt::printStackTrace();
    abort();
  }

  // TODO: We need to consider multi-gpu device support and use cases where
  // multiple threads (parallel constructs) can drive multiple gpus.

  // There have been several cases where debugging and performance regressions
  // have tripped us up on multi-gpu systems. To help out in these cases we
  // provide a path to pick the device id via the environment.
  if (std::optional<int> id = kitrt::envLookup<int>("KITHIP_DEVICE_ID")) {
    assert(id >= 0 && "kitrt[hip]: KITHIP_DEVICE_ID is invalid");
    assert(id < deviceCount() && "kitrt[hip]: KITHIP_DEVICE_ID is in range");
    LOG("env override, using device: %d.\n", id);
    setDeviceID(*id);
  } else {
    LOG("using default device");
    setDeviceID(0); // Default is always the first device.
  }

  HIP_SAFE_CALL(hipSetDevice(deviceID()));
  HIP_SAFE_CALL(hipGetDeviceProperties(&rt_info.props, deviceID()));
  if (kitrt::gctx.verbose)
    __kithip_dump_dev_properties(rt_info.props);

  // Apparently this is the only way to determine if the device is GCN or not.
  rt_info.isGCN = std::string_view(rt_info.props.gcnArchName).find("gfx9") == 0;

  if (not supportsManagedMemory()) {
    fprintf(stderr,
            "kitrt[hip]: device/system does not support managed memory!\n");
    fprintf(stderr, "  kitsune does not support this platform.\n");
    abort();
  }

  if (std::optional<int> tpb =
          kitrt::envLookup<int>("KITHIP_THREADS_PER_BLOCK")) {
    __kithip_set_threads_per_blk(*tpb);
    if (kitrt::gctx.verbose)
      fprintf(stderr, "  kithip: threads/block: %d\n", *tpb);
  }

  if (std::optional<int> tpb =
          kitrt::envLookup<int>("KITHIP_MAX_THREADS_PER_BLOCK")) {
    __kithip_set_max_threads_per_blk(*tpb);
    if (kitrt::gctx.verbose)
      fprintf(stderr, "  kithip: max threads/block: %d\n", *tpb);
  }

  // We should be good to go...
  setInitialized(true);

  LOG("Initialized Kitsune runtime (hip)");
}

extern "C" void __kithip_finalize(void) {
  using namespace kithip_rt;
  if (not isInitialized()) {
    LOG("Cannot finalize runtime. Not initialized");
    return;
  }

  LOG("Finalizing Kitsune runtime (hip)");

  __kithip_destroy_thread_streams();
  __kitrt_destroy_memory_map(__kithip_mem_destroy);

  // FIXME: Figure out what why hipDeviceReset() segfaults. There is a probably
  // some resource cleanup that is not being done correctly.
  // HIP_SAFE_CALL(hipDeviceReset());

  setInitialized(false);

  LOG("Finalized Kitsune runtime (hip)");

  // Finalize the components of Kitsune's runtime that are shared by the
  // tapir-target-specific components.
  __kitrt_finalize();
}

extern "C" uint64_t __kithip_num_cus(void) {
  // On GCN, the number of compute units reported is the actual number that are
  // available, while on other architectures, the reported value must be
  // multiplied by 2 to get the actual number available. Because that is
  // definitely reasonable. This is being tracked here:
  //
  //     https://github.com/rocm/rocm/issues/6407
  //
  int mps = kithip_rt::rt_info.props.multiProcessorCount;
  if (kithip_rt::rt_info.isGCN)
    return mps;
  return mps * 2;
}

// These are declarations from ROCm's "internal" headers (internal in the sense
// that they are in headers that are part of the ROCm installation, but they
// don't look like they are intended for the average user)
extern "C" void **__hipRegisterFatBinary(const void *data);
extern "C" void __hipUnregisterFatBinary(void **modules);
extern "C" void __hipRegisterVar(void **modules, void *var, char *hostVar,
                                 char *deviceVar, int ext, size_t size,
                                 int constant, int global);
extern "C" void __hipRegisterManagedVar(void *hipModule, void **pointer,
                                        void *init_value, const char *name,
                                        size_t size, unsigned align);

/// Register a global variable containing device code. We intentionally do not
/// refer to this as a "fat binary" because, at the time of writing, the code is
/// for a single device architecture. This will nearly always be called in the
/// global constructor for Kitsune's runtime.
///
/// \param data Pointer to the global containing the device code.
/// \returns An opaque handle that should be used to register global variables
///          used in the device, and potentially other things.
extern "C" void **__kithip_register_devcode(void *data) {
  return __hipRegisterFatBinary(data);
}

/// Unregister all device code that was previously registered in a call to
/// __kithip_register_devcode.
///
/// \param handle An opaque handle returned by __kithip_register_devcode.
extern "C" void __kithip_unregister_devcode(void **handle) {
  return __hipUnregisterFatBinary(handle);
}

/// Register a global variable that is present in the device code that was
/// previously registered with a call to __kithip_register_devcode. This will
/// nearly always be called from a global constructor for Kitsune's runtime.
///
/// \param handle An opaque handle returned by __kithip_register_devcode.
/// \param hostAddr The address of the corresponding "shadow" variable on the
///                 host. The device-side global variable will be initialized
///                 with the value on the host.
/// \param hostName Name of the global in host code.
/// \param deviceName Name of the global variable in device code.
/// \param size Size, in the bytes, of the global variable.
/// \param isExternal Is the global variable externally visible.
/// \param isConstant Is The global variable constant.
extern "C" void __kithip_register_global(void **handle, void *hostAddr,
                                         char *hostName, char *deviceName,
                                         size_t size, int isExternal,
                                         int isConstant) {
  // Per the documentation, the last argument must always be zero.
  return __hipRegisterVar(handle, hostAddr, hostName, deviceName, isExternal,
                          size, isConstant, /*global=*/0);
}

/// Register a global variable that is present in the device code that was
/// previously registered with a call to __kitcuda_register_devcode. This will
/// allocate space for the global variable in UVM and return a pointer to that
/// allocated memory via the \p newAddr out variable. This will nearly always
/// always be called from a global constructor for Kitsune's runtime.
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
/// \param isConstant Is the global variable constant.
extern "C" void
__kithip_register_global_managed(void **handle, void **newAddr, void *hostAddr,
                                 const char *deviceName, size_t size, int align,
                                 int isExternal, int isConstant) {
  // In the declaration for __hipRegisterManagedVar, the third argument is
  // named initial_value. One would expect to be simply a pointer to the global
  // on the host since it would contain the initial value.
  FATAL("TODO: Check __kithip_register_global_managed for correctness");

  // FIXME?: Is it correct for the last argument to be zero? This is the case
  // when registering normal (those that are not allocated in managed memory)
  // globals.
  return __hipRegisterManagedVar(handle, newAddr, hostAddr, deviceName, size,
                                 align);
}
