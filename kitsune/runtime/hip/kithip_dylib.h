/*
 *===---- Kitsune HIP dynamic library helpers -----------------------------===
 *
 * Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2021, 2023. Los Alamos National Security, LLC. This software 
 *  was produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 *
 *===----------------------------------------------------------------------===
 */
#ifndef __KITRT_HIP_DL_H__
#define __KITRT_HIP_DL_H__

#include "dlutils.h"

/**
 * Load the cuda dynamic library symbols we need for the kitsune
 * runtime.  This requires the cuda library (libcuda.so) to be
 * present in your dynamic library search path.
 */
extern bool __kithip_load_symbols();

/* Initialization and device-wide entry points */ 
DECLARE_DLSYM(hipInit);
DECLARE_DLSYM(hipGetDeviceCount);
DECLARE_DLSYM(hipGetDevice);
DECLARE_DLSYM(hipSetDevice);
DECLARE_DLSYM(hipDeviceGetAttribute);
DECLARE_DLSYM(hipGetDeviceProperties);
DECLARE_DLSYM(hipDeviceReset);
DECLARE_DLSYM(hipDeviceSynchronize);

/* Context management */

/* Stream management */
DECLARE_DLSYM(hipStreamCreate);
DECLARE_DLSYM(hipStreamCreateWithFlags);
DECLARE_DLSYM(hipStreamDestroy);
DECLARE_DLSYM(hipStreamSynchronize);

/* Kernel launching, fat binary, module related */
DECLARE_DLSYM(hipModuleLoadData);
DECLARE_DLSYM(hipModuleGetGlobal);
DECLARE_DLSYM(hipModuleGetFunction);
DECLARE_DLSYM(hipLaunchKernel);
DECLARE_DLSYM(hipModuleLaunchKernel);
DECLARE_DLSYM(hipModuleOccupancyMaxPotentialBlockSize);

/* Memory management and movement */
DECLARE_DLSYM(hipMallocManaged);
DECLARE_DLSYM(hipFree);
DECLARE_DLSYM(hipMemAdvise);
DECLARE_DLSYM(hipMemRangeGetAttribute);
DECLARE_DLSYM(hipPointerGetAttribute);
DECLARE_DLSYM(hipPointerGetAttributes);
DECLARE_DLSYM(hipMemcpy);
DECLARE_DLSYM(hipMemcpyHtoD);
DECLARE_DLSYM(hipMemsetD8Async);
DECLARE_DLSYM(hipMemPrefetchAsync);

/* Error handling */
DECLARE_DLSYM(hipGetErrorName);
DECLARE_DLSYM(hipGetErrorString);

#endif
