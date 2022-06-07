//
//===- llvm-cuda.cpp - Kitsune ABI runtime target CUDA support    ---------===//
//
// TODO:
//     - Need to update LANL/Triad Copyright notice.
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

#include "kitrt-debug.h"
#include "llvm-cuda.h"
#include "kitrt-cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <stdbool.h>


// Has the runtime been initialized (successfully)?
static bool _kitrtIsInitialized = false;
static bool _kitrtEnableTiming = false;
static unsigned _kitrtDefaultThreadsPerBlock = 256;
static unsigned _kitrtDefaultBlocksPerGrid = 0;
static bool _kitrtUseCustomLaunchParameters = false;
static CUdevice  _kitrtCUdevice = -1;
static CUcontext _kitrtCUcontext = nullptr;

#define declare(name) extern decltype(name) *name##_p = NULL;
declare(cuInit);
declare(cuDeviceGetCount);
declare(cuDeviceGet);

declare(cuCtxCreate);
declare(cuDevicePrimaryCtxRetain);
declare(cuCtxDestroy);
declare(cuCtxSetCurrent);
declare(cuCtxPushCurrent);
declare(cuCtxPopCurrent);
declare(cuCtxGetCurrent);
declare(cuStreamCreate);
//declare(cuStreamDestroy_v2);
declare(cuStreamSynchronize);
declare(cuLaunchKernel);
declare(cuEventCreate);
declare(cuEventRecord);
declare(cuEventSynchronize);
declare(cuEventElapsedTime);
declare(cuEventDestroy);
declare(cuGetErrorName);
declare(cuGetErrorString);
declare(cuModuleLoadDataEx);
declare(cuModuleLoadData);
declare(cuModuleLoadFatBinary);
declare(cuModuleGetFunction);
declare(cuModuleUnload);

declare(cuMemAllocManaged);
declare(cuMemFree);
declare(cuMemPrefetchAsync);
declare(cuMemAdvise);
declare(cuPointerGetAttribute);
declare(cuDeviceGetAttribute);
declare(cuCtxSynchronize);
declare(cuModuleGetGlobal);
declare(cuMemcpy);
declare(cuMemcpyHtoD);

// TODO: We could make these checks optional for optimized builds.

#define CU_SAFE_CALL(x)                                      \
  {                                                          \
    CUresult result = x;                                     \
    if (result != CUDA_SUCCESS) {                            \
      const char *msg;                                       \
      cuGetErrorName_p(result, &msg);                        \
      fprintf(stderr, "kitrt %s:%d:\n", __FILE__, __LINE__); \
      fprintf(stderr, "  %s failed ('%s')\n", #x, msg);      \
      cuGetErrorString_p(result, &msg);                      \
      fprintf(stderr, "  error: '%s'\n", msg);               \
      exit(1);                                               \
    }                                                        \
  }

typedef std::map<void *, size_t> KitRTAllocMap;
static KitRTAllocMap _kitrtAllocMap;

static void __kitrt_registerMemAlloc(void *addr, size_t size) {
  assert(addr != nullptr && "unexpected null pointer!");
  assert(_kitrtAllocMap.find(addr) == _kitrtAllocMap.end() && "insertion of existing mem alloc pointer!");
  _kitrtAllocMap[addr] = size;
}

static size_t __kitrt_getMemAllocSize(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::const_iterator cit = _kitrtAllocMap.find(addr);
  if (cit != _kitrtAllocMap.end())
    return cit->second;
  else
    return 0;
}

static bool __kitrt_unregisterMemAlloc(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::iterator it = _kitrtAllocMap.find(addr);
  if (it != _kitrtAllocMap.end()) {
    _kitrtAllocMap.erase(it);
    return true;
  } else {
    return false;
  }
}

static bool __kitrt_load_dlsyms() {

  #define DLSYM_LOAD(FN)                                       \
    if (! (FN##_p = (decltype(FN)*)dlsym(dlHandle, #FN))) {    \
      fprintf(stderr, "kitrt: failed to load dlsym 'FN##'.");  \
      return false;}

  static void *dlHandle = nullptr;
  if (dlHandle)
    return true;

  if (dlHandle = dlopen("libcuda.so", RTLD_LAZY)) {
    DLSYM_LOAD(cuInit);
    DLSYM_LOAD(cuGetErrorName);
    DLSYM_LOAD(cuGetErrorString);
    DLSYM_LOAD(cuDeviceGetCount);
    DLSYM_LOAD(cuDeviceGet);
    DLSYM_LOAD(cuDevicePrimaryCtxRetain);
    DLSYM_LOAD(cuCtxCreate);
    DLSYM_LOAD(cuCtxDestroy);
    DLSYM_LOAD(cuCtxSetCurrent);
    DLSYM_LOAD(cuCtxPushCurrent);
    DLSYM_LOAD(cuCtxPopCurrent);
    DLSYM_LOAD(cuCtxGetCurrent);

    //DLSYM_LOAD(cuCtxCreate_v2);
    //DLSYM_LOAD(cuCtxDestroy_v2);
    DLSYM_LOAD(cuMemAllocManaged);
    DLSYM_LOAD(cuMemFree);
    DLSYM_LOAD(cuMemPrefetchAsync);
    DLSYM_LOAD(cuMemAdvise);

    DLSYM_LOAD(cuModuleLoadData);
    DLSYM_LOAD(cuModuleLoadDataEx);
    DLSYM_LOAD(cuModuleLoadFatBinary);
    DLSYM_LOAD(cuModuleGetFunction);
    DLSYM_LOAD(cuModuleGetGlobal);
    DLSYM_LOAD(cuModuleUnload);

    DLSYM_LOAD(cuStreamCreate);
    //DLSYM_LOAD(cuStreamDestroy_v2);
    DLSYM_LOAD(cuStreamSynchronize);
    DLSYM_LOAD(cuLaunchKernel);

    DLSYM_LOAD(cuEventCreate);
    DLSYM_LOAD(cuEventRecord);
    DLSYM_LOAD(cuEventSynchronize);
    DLSYM_LOAD(cuEventElapsedTime);
    DLSYM_LOAD(cuEventDestroy);



    DLSYM_LOAD(cuPointerGetAttribute);
    DLSYM_LOAD(cuDeviceGetAttribute);
    DLSYM_LOAD(cuCtxSynchronize);

    DLSYM_LOAD(cuMemcpy);
    DLSYM_LOAD(cuMemcpyHtoD);
    return true;
  } else {
    fprintf(stderr, "kitrt: Failed to load CUDA dynamic library.\n");
    fprintf(stderr, "kitrt: Is CUDA in your LD_LIBRARY_PATH?\n");
    return false;
  }
}

extern "C"
bool __kitrt_cuInit() {
  if (_kitrtIsInitialized)
    return true;
  fprintf(stderr, "called __kitrt_cuInit()...\n");

  if (!__kitrt_load_dlsyms()) {
    fprintf(stderr, "kitrt: Unable to resolve dynamic symbols for CUDA.\n");
    fprintf(stderr, "kitrt: Check your enviornment settings and CUDA installation.\n");
    fprintf(stderr, "kitrt: aborting...\n");
    abort();
  }

  int deviceCount = 0;

  CU_SAFE_CALL(cuInit_p(0 /*, __CUDA_API_VERSION*/));
  CU_SAFE_CALL(cuDeviceGetCount_p(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "kitrt: no CUDA devices were found. aborting...\n");
    exit(1);
  }
  CU_SAFE_CALL(cuDeviceGet_p(&_kitrtCUdevice, 0));
  CU_SAFE_CALL(cuDevicePrimaryCtxRetain_p(&_kitrtCUcontext, _kitrtCUdevice));
  // NOTE: It seems we have to explicitly set the context but that seems
  // to be different than what the driver API docs suggest...
  CU_SAFE_CALL(cuCtxSetCurrent_p(_kitrtCUcontext));
  _kitrtIsInitialized = true;
  return _kitrtIsInitialized;
}

extern "C" void __kitrt_cuSetCustomLaunchParameters(unsigned BlocksPerGrid,
                                                    unsigned ThreadsPerBlock) {
  _kitrtUseCustomLaunchParameters = true;
  _kitrtDefaultBlocksPerGrid = BlocksPerGrid;
  _kitrtDefaultThreadsPerBlock = ThreadsPerBlock;
}

extern "C" void __kitrt_cuSetDefaultThreadsPerBlock(unsigned ThreadsPerBlock) {
  _kitrtDefaultThreadsPerBlock = ThreadsPerBlock;
}

extern "C" void __kitrt_cuEnableEventTiming() {
  _kitrtEnableTiming = true;
}

extern "C" void __kitrt_cuDisableEventTiming() {
  _kitrtEnableTiming = false;
}

extern "C" void __kitrt_cuToggleEventTiming() {
  _kitrtEnableTiming = _kitrtEnableTiming ? false : true;
}

extern "C" void* __kitrt_cuCreateEvent() {
  CUevent e;
  CU_SAFE_CALL(cuEventCreate_p(&e, CU_EVENT_DEFAULT));
  return (void*)e;
}

extern "C" void __kitrt_cuRecordEvent(void *E) {
  assert(E && "__kitrt_cuRecordEvent() null event!");
  CU_SAFE_CALL(cuEventRecord_p((CUevent)E, 0));
}

extern "C" void __kitrt_cuSynchronizeEvent(void *E) {
  assert(E && "__kitrt_cuSynchronizeEvent() null event!");
  CU_SAFE_CALL(cuEventSynchronize_p((CUevent)E));
}

extern "C" void __kitrt_cuDestroyEvent(void *E) {
  assert(E && "__kitrt_cuEventDestory() null event!");
  CU_SAFE_CALL(cuEventDestroy_p((CUevent)E));
}

extern "C" float __kitrt_cuElapsedEventTime(void *start, void *stop) {
  assert(start && "__kitrt_cuElapsedEventTime() null starting event");
  float msecs;
  CU_SAFE_CALL(cuEventElapsedTime_p(&msecs, (CUevent)start, (CUevent)stop));
  return(msecs/1000.0f);
}

extern "C" bool __kitrt_cuIsMemManaged(void *vp) {
  assert(vp && "__kitrt_cuIsMemManaged() null data pointer!");
  CUdeviceptr devp = (CUdeviceptr)vp;

  /* For a tad bit of flexiblity we don't wrap this call in a
   * safe call -- we want to return false if the given pointer
   * is "junk" as far as CUDA is concerned.
   */
  unsigned int is_managed;
  CUresult r = cuPointerGetAttribute_p(&is_managed,
                                       CU_POINTER_ATTRIBUTE_IS_MANAGED, devp);
  if (r == CUDA_SUCCESS)
    return is_managed ? true : false;
  else
    return false;
}

extern "C" void __kitrt_cuMemPrefetchIfManaged(void *vp, size_t size) {
  if (__kitrt_cuIsMemManaged(vp))
    __kitrt_cuMemPrefetchAsync(vp, size);
}

extern "C" void __kitrt_cuMemPrefetchAsync(void *vp, size_t size) {
  assert(vp && "__kitrt_cuMemPrefetchAsync() null data pointer!");
  CUdeviceptr devp = (CUdeviceptr)vp;
  CU_SAFE_CALL(cuMemPrefetchAsync_p(devp, size, _kitrtCUdevice, NULL));
}

extern "C" void __kitrt_cuMemPrefetch(void *vp) {
  assert(vp && "__kitrt_cmMemPrefetch() null data pointer!");
  size_t size = __kitrt_getMemAllocSize(vp);
  // TODO: In theory -- but perhaps not practice -- we should only get a
  // non-zero size back for data that has been allocated as managed memory.
  // So only prefetch with that in mind...
  if (size > 0)
    __kitrt_cuMemPrefetchAsync(vp, size);
  else
    fprintf(stderr, "__kitrt: warning, prefetch requested but referenced pointer not found?\n");
}

extern "C"
void *__kitrt_cuMemAllocManaged(size_t size) {
  fprintf(stderr, "called __kitrt_cuMemAllocManaged()...\n");
  (void)__kitrt_cuInit();
  CUdeviceptr devp;
  CU_SAFE_CALL(cuMemAllocManaged_p(&devp, size, CU_MEM_ATTACH_HOST));
  __kitrt_registerMemAlloc((void*)devp, size);
  return (void *)devp;
}

extern "C" void __kitrt_cuMemFree(void *vp) {
    assert(vp && "__kitrt_cuMemFree() null data pointer!");
    if (__kitrt_unregisterMemAlloc(vp)) {
      // TODO: we shold probably do something more here than ignore a 'false' unregister call.
      CUdeviceptr devp = (CUdeviceptr)vp;
      CU_SAFE_CALL(cuMemFree_p(devp));
    }
}


extern "C"
void __kitrt_cuAdviseRead(void *vp, size_t size) {
  CUdeviceptr devp = (CUdeviceptr)vp;
  CU_SAFE_CALL(cuMemAdvise_p(devp, size, CU_MEM_ADVISE_SET_READ_MOSTLY, _kitrtCUdevice));
  CU_SAFE_CALL(cuMemAdvise_p(devp, size, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, _kitrtCUdevice));
}


extern "C"
void __kitrt_cuMemcpySymbolToDevice(void *hostPtr, void *devSym, size_t size) {
  (void)__kitrt_cuInit();
  assert(devSym != nullptr &&
         "__kitrt_cuMemcpySymbolToDevice() -- null device symbol pointer!");
  assert(hostPtr != nullptr &&
         "__kitrt_cuMemcpySymbolToDevice() -- null host symbol pointer!");
  assert(size != 0 &&
         "__kitrt_cuMemcpySymbolToDevice() -- requested a 0 byte copy!");
  CUdeviceptr DevPtr = (CUdeviceptr)devSym;
  CU_SAFE_CALL(cuMemcpyHtoD_p(DevPtr, hostPtr, size));
}

static void __kitrt_cuGetLaunchParameters(size_t &threadsPerBlock,
                                          size_t &blocksPerGrid,
                                          size_t numElements) {
  if (_kitrtUseCustomLaunchParameters) {
    threadsPerBlock = _kitrtDefaultThreadsPerBlock;
    blocksPerGrid = _kitrtDefaultBlocksPerGrid;
    // reset after the launch.
    _kitrtUseCustomLaunchParameters = false;
  } else {
    int warpSize;
    CU_SAFE_CALL(cuDeviceGetAttribute_p(&warpSize,
                                        CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                        _kitrtCUdevice));
    unsigned blockSize = 4 * warpSize;
    //assert(numElements % blockSize == 0);
    // TODO: There is likely quite a bit of improvement to be had here
    // in terms of connecting the compiler and runtime components for a
    // better set of launch parameters.   Might be best to combine code
    // analysis from codegen time with runtime calls to bind parameters
    // to a specific launch...
    threadsPerBlock = _kitrtDefaultThreadsPerBlock;
    blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  }
}


extern "C"
void *__kitrt_cuCreateFBModule(const void *fatBin) {
  assert(fatBin && "request to create module from null fatbinary!");
  CUmodule module;
  CU_SAFE_CALL(cuModuleLoadData_p(&module, fatBin));
  //CU_SAFE_CALL(cuModuleLoadFatBinary_p(&module, fatBin));
  printf("global symbol module: %p\n", (void*)module);
  return (void*)module;
}

extern "C"
void *__kitrt_cuGetGlobalSymbol(const char *SN, void *CM) {
  assert(SN && "null symbol name (SN)!");
  assert(CM && "null (opaque) CUDA module");
  CUmodule Module = (CUmodule)CM;
  // NOTE: The device poitner and size ('bytes') parameters for
  // cuModuleGetGlobal are optional.  To simplify our code gen
  // work we ignore the size parameter (which is NULL below).
  CUdeviceptr DevPtr;
  size_t nbytes;
  fprintf(stderr, "kitrt_cuGetGlobalSymbol '%s'\n", SN);
  CU_SAFE_CALL(cuModuleGetGlobal_p(&DevPtr, &nbytes, Module, SN));

  //CU_SAFE_CALL(cuCtxPopCurrent_p(&_kitrtCUcontext));
  return (void*)DevPtr;
}


extern "C"
void *__kitrt_cuLaunchModuleKernel(void *mod,
                                   const char *kernelName,
                                   void **fatBinArgs,
                                   uint64_t numElements) {
  // TODO: we probably want to calculate the launch compilers during code
  // generation (when we have some more information about the actual kernel
  // code -- in that case, we should pass launch parameters to this call.
  size_t threadsPerBlock, blocksPerGrid;
  __kitrt_cuGetLaunchParameters(threadsPerBlock, blocksPerGrid, numElements);

  CUfunction kFunc;
  CUmodule module = (CUmodule)mod;
  CU_SAFE_CALL(cuModuleGetFunction_p(&kFunc, module, kernelName));

  CUevent start, stop;
  if (_kitrtEnableTiming) {
    // Recall that we have to take a bit of care about how we time the
    // launched kernel's execution time.  The problme with using host-device
    // synchronization points is that they can potentially stall the entire
    // GPU pipeline, which we want to avoid to enable asynchronous data
    // movement and the execution of other kernels on the GPU.
    //
    // A nice overview for measuring performance in CUDA:
    //
    //   https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    //
    // TODO: What event creation flags do we really want here?   See:
    //
    //   https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html
    //
    cuEventCreate_p(&start, CU_EVENT_DEFAULT);
    cuEventCreate_p(&stop, CU_EVENT_DEFAULT);
    cuEventRecord_p(start, 0);
  }

  CU_SAFE_CALL(cuLaunchKernel_p(kFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1,
                                1, 0, nullptr, fatBinArgs, NULL));

  if (_kitrtEnableTiming) {
    cuEventRecord_p(stop, 0);
    cuEventSynchronize_p(stop);
    float msecs = 0;
    cuEventElapsedTime_p(&msecs, start, stop);
    printf("%.8lg\n", msecs / 1000.0);
    cuEventDestroy_p(start);
    cuEventDestroy_p(stop);
  }

  return nullptr;
}


extern "C"
void *__kitrt_cuStreamLaunchFBKernel(const void *fatBin,
                                     const char *kernelName,
                                     void **fatBinArgs,
                                     uint64_t numElements) {
  assert(fatBin && "request to launch null fat binary image!");
  assert(kernelName && "request to launch kernel w/ null name!");

  // TODO: we probably want to calculate the launch compilers during code
  // generation (when we have some more information about the actual kernel
  // code -- in that case, we should pass launch parameters to this call.
  size_t threadsPerBlock, blocksPerGrid;
  __kitrt_cuGetLaunchParameters(threadsPerBlock, blocksPerGrid, numElements);

  // TODO: We need a better path here for binding and tracking
  // allcoated CUDA resources -- as it stands we will "leak"
  // modules, streams, and functions...
  CUmodule module;
  CU_SAFE_CALL(cuModuleLoadData_p(&module, fatBin));
  CUfunction kFunc;
  CU_SAFE_CALL(cuModuleGetFunction_p(&kFunc, module, kernelName));
  CUstream stream = nullptr;
  CU_SAFE_CALL(cuStreamCreate_p(&stream, 0));

  CUevent start, stop;
  if (_kitrtEnableTiming) {
    // Recall that we have to take a bit of care about how we time the
    // launched kernel's execution time.  The problme with using host-device
    // synchronization points is that they can potentially stall the entire
    // GPU pipeline, which we want to avoid to enable asynchronous data
    // movement and the execution of other kernels on the GPU.
    //
    // A nice overview for measuring performance in CUDA:
    //
    //   https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    //
    // TODO: What event creation flags do we really want here?   See:
    //
    //   https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html
    //
    cuEventCreate_p(&start, CU_EVENT_DEFAULT);
    cuEventCreate_p(&stop, CU_EVENT_DEFAULT);
    cuEventRecord_p(start, stream);
  }

  CU_SAFE_CALL(cuLaunchKernel_p(kFunc, blocksPerGrid, 1, 1, threadsPerBlock,
                                1, 1, 0, stream, fatBinArgs, NULL));

  if (_kitrtEnableTiming) {
    cuEventRecord_p(stop, stream);
    cuEventSynchronize_p(stop);
    float msecs = 0;
    cuEventElapsedTime_p(&msecs, start, stop);
    printf("%.8lg\n", msecs / 1000.0);
    cuEventDestroy_p(start);
    cuEventDestroy_p(stop);
  }

  return (void *)stream;
}

extern "C" void *__kitrt_cuLaunchFBKernel(const void *fatBin,
                                          const char *kernelName,
                                          void **fatBinArgs,
                                          uint64_t numElements) {
  assert(fatBin && "request to launch null fat binary image!");
  assert(kernelName && "request to launch kernel w/ null name!");

  // TODO: we probably want to calculate the launch compilers during code
  // generation (when we have some more information about the actual kernel
  // code -- in that case, we should pass launch parameters to this call.
  size_t threadsPerBlock, blocksPerGrid;
  __kitrt_cuGetLaunchParameters(threadsPerBlock, blocksPerGrid, numElements);

  // TODO: We need a better path here for binding and tracking
  // allcoated CUDA resources -- as it stands we will "leak"
  // modules, streams, and functions...
  CUmodule module;
  CU_SAFE_CALL(cuModuleLoadData_p(&module, fatBin));
  CUfunction kFunc;
  CU_SAFE_CALL(cuModuleGetFunction_p(&kFunc, module, kernelName));

  CUevent start, stop;
  if (_kitrtEnableTiming) {
    // Recall that we have to take a bit of care about how we time the
    // launched kernel's execution time.  The problme with using host-device
    // synchronization points is that they can potentially stall the entire
    // GPU pipeline, which we want to avoid to enable asynchronous data
    // movement and the execution of other kernels on the GPU.
    //
    // A nice overview for measuring performance in CUDA:
    //
    //   https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    //
    // In the non-stream kernel launch we use the default stream and thus
    // some behaviors related to timing could vary...
    //
    // TODO: What event creation flags do we really want here?   See:
    //
    //   https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html
    //
    cuEventCreate_p(&start, CU_EVENT_DEFAULT);
    cuEventCreate_p(&stop, CU_EVENT_DEFAULT);

    // Kick off an event prior to kernel launch...
    cuEventRecord_p(start, 0);
  }

  CU_SAFE_CALL(cuLaunchKernel_p(kFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1,
                                1, 0, nullptr, fatBinArgs, NULL));

  if (_kitrtEnableTiming) {
    // This might seem counterintuitive at first glance but we record the
    // 'stop' event prior to synchronization...
    // priot to sychronizing execution.
    cuEventRecord_p(stop, 0);
    cuEventSynchronize_p(stop);

    float msecs = 0;
    cuEventElapsedTime_p(&msecs, start, stop);
    printf("%.8lg\n", msecs / 1000.0);
    cuEventDestroy_p(start);
    cuEventDestroy_p(stop);
  }

  return nullptr;
}


// TODO: This call currently uses a hard-coded kernel name in the
// launch.  Once that is fixed in the ABI code, we can repalce this
// call with the FB ("fat binary") launch call above...
extern "C"
void *__kitrt_cuLaunchELFKernel(const void *elf,
                                void **args,
                                size_t numElements) {
  CUmodule module;
  CUfunction kernel;
  CU_SAFE_CALL(cuModuleLoadDataEx_p(&module, elf, 0, 0, 0));
  CU_SAFE_CALL(cuModuleGetFunction_p(&kernel, module, "kitsune_kernel"));
  CUstream stream = nullptr;
  CU_SAFE_CALL(cuStreamCreate_p(&stream, 0));
  size_t threadsPerBlock, blocksPerGrid;
  __kitrt_cuGetLaunchParameters(threadsPerBlock, blocksPerGrid, numElements);
  CU_SAFE_CALL(cuLaunchKernel_p(kernel, blocksPerGrid, 1, 1, // grid dim
                                threadsPerBlock, 1, 1,       // block dim
                                0, stream,    // shared mem and stream
                                args, NULL)); // arguments
  return stream;
}


extern "C"
void *__kitrt_cuLaunchKernel(llvm::Module &m, void **args, size_t n) {
  std::string ptx = __kitrt_cuLLVMtoPTX(m, _kitrtCUdevice);
  void *elf = __kitrt_cuPTXtoELF(ptx.c_str());
  assert(elf && "failed to compile PTX kernel to ELF image!");
  return __kitrt_cuLaunchELFKernel(elf, args, n);
}

extern "C"
void __kitrt_cuStreamSynchronize(void *vs) {
  if (_kitrtEnableTiming)
    return; // TODO: In theory we sync'ed at kernel launch to time
            // the execution.  So, we don't need this... ???

  // TODO: we can probably just stream sync on the default
  // stream here but let's be a bit more 'precise' (not sure
  // if it matters so we should look into it more).
  if (vs != nullptr) {
    CU_SAFE_CALL(cuStreamSynchronize_p((CUstream)vs));
  } else {
    CU_SAFE_CALL(cuStreamSynchronize_p(0));
  }
}

extern "C"
void __kitrt_cuCheckCtxState() {
  if (_kitrtIsInitialized) {
    CUcontext c;
    CU_SAFE_CALL(cuCtxGetCurrent_p(&c));
    if (c != _kitrtCUcontext) {
      fprintf(stderr, "kitrt: warning! current cuda context mismatch!\n");
    } else {
      fprintf(stderr, "kitrt: ok - current cuda context as expected!\n");
    }
  } else {
    fprintf(stderr,
	    "kitrt: context check encountered uninitialized CUDA state!\n");
  }
}

