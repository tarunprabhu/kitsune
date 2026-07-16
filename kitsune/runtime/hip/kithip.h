//===- kithip.h - HIP runtime interface -------------------------*- C++ -*-===//
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

#ifndef __KITHIP_H_
#define __KITHIP_H_

#include <stdint.h>
#include <stdlib.h>

#include "kitrt.h"

#define __HIP_DISABLE_CPP_FUNCTIONS__ // skip extra c++ cruft
// we're only interested in AMD GPUs and HIP (no CUDA).
#define __HIP_PLATFORM_AMD__ 1
#define __HIP_PLATFORM_HCC__ 1
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

/**
 * Initialize the HIP portion of the Kitsune runtime library. The
 * initialization process will load a number of dynamic entry points, implement
 * core parts of the runtime that are independent of HIP, set an active HIP
 * device, set the primary context set the context.
 *
 *   - Multiple calls to the function will guard against re-initialization if it
 *     was previously successful.
 *
 *   - This call is not thread safe.
 *
 * There are a number of environment variables that can tweak the behavior of
 * the runtime:
 *
 *   - **KITHIP_THREADS_PER_BLOCK**: Number of threads per block of the kernel
 *     launch. This number has an internal default (currently 256). This is a
 *     global setting and will apply to all kernel launches.
 *
 *   - **KITHIP_DEVICE_ID**: Select a specific GPU device to use. This is
 *     intended to allow experimentation across different GPUs within a single
 *     system. The runtime currently only supports a single GPU and this will
 *     default to the first GPU in the system if left unset.
 *
 * Applications should call `__kithip_finalize()` at program exit.
 *
 **/
void __kithip_initialize(void);

/**
 * Enable the use of XNACK for an executing program. This call should be made
 * prior to runtime initialization.  It has the side effect of setting the XNACK
 * environment variable that is specific to the HIP/ROCm feature set.
 *
 * TODO: document more about this as it is opaque.
 */
void __kithip_enable_xnack(void);

/**
 * Load the requried HIP dynamic symbols for use by the runtime.
 */
bool __kithip_load_dlsyms(void);

/**
 * Free and release any resources used by the Kitsune HIP runtime component.
 * This call should be made at program termination to clean up and release any
 * bound GPUs.
 *
 * Any calls into the Kitsune HIP runtime after this call is made are likely to
 * fail with either an assertion or an internal HIP error.
 *
 * This call is not thread safe.
 */
void __kithip_finalize(void);

/**
 * Return `true` if the provided pointer has been allocated as a block of
 * managed memory, `false` otherwise.
 *
 * @param vp The pointer to query.
 */
bool __kithip_is_mem_managed(void *vp);

/**
 * Allocate the given number of bytes of managed heap memory. This memory
 * allocation, also referred to as *unified virtual memory*, will have its pages
 * managed by the OS and be paged across CPU and GPU memories as accessed.
 *
 * It is important that this call be used instead of HIP's managed memory
 * allocation function. Kitsune's runtime will track this allocation to enable
 * automatic prefetching of data when it is part of the managed allocation.
 *
 * @param num_bytes The number of bytes to allocate as part of this request
 *
 * This call will return NULL on failure or a pointer to the allocated managed
 * memory pointer on success.
 */
[[gnu::malloc]] void *__kithip_mem_alloc_managed(size_t num_bytes);

/**
 * Allocate the given number of bytes of managed heap memory. This memory
 * allocation, also referred to as *unified virtual memory*, will have its pages
 * managed by the OS and be paged across CPU and GPU memories as accessed. This
 * call matches the behavior of the system `calloc()` call and the entire
 * allocated memory will be set to zero.
 *
 * With managed memory it is important to note that this function uses the CPU
 * to zero-initialize the allocation - this means the memory will become fully
 * host-side resident upon return.
 *
 * @param count The number of elements to allocate.
 * @param elem_size_in_bytes The size, in bytes, of a single element.
 *
 * @todo Make a path for device-side initialization.
 */
[[gnu::malloc]] void *__kithip_mem_calloc_managed(size_t count,
                                                  size_t elemsize);

/**
 * Change the size of an existing managed memory allocation (similar to the
 * `realloc()` system call). The current contents of the previously allocated
 * block will be unchanged in the range from the start of provided address up to
 * minimum of the old and new sizes. If the new size is greater than the
 * previous allocation, the additional space will be uninitialized. A few notes
 * about behavior:
 *
 *   - If the passed in pointer is NULL this will be equivalent to allocating a
 *     new managed block of memory of the given size.
 *
 *   - If the requested size is zero, and the pointer is not NULL, the call is
 *     equivalent to freeing the provided pointer.
 *
 *   - Like traditional system calls, the provided pointer must have been
 *     allocated by the kitsune runtime routines. The allocation will fail if a
 *     non-kitsune allocation is referenced.
 *
 * @param ptr The pointer to *realloc* space for.
 * @param size The size, in bytes, of the new allocation.
 *
 * This call is currently not thread safe.
 *
 * @todo Look at thread safe implementation.
 */
[[gnu::malloc]] void *__kithip_mem_realloc_managed(void *ptr, size_t size);

/**
 * Free the given managed memory allocation. The allocation referred to by `ptr`
 * must have been previously allocated with one of the kitsune managed memory
 * allocations. Failure to free such memory using this routine will result in
 * an inconsistent runtime state that will most certainly lead to a runtime leak
 * in the best case and a fatal crash or memory corruption in the worst cases.
 *
 * @param ptr A pointer to a previous Kitsune managed memory allocation.
 *
 * This call is currently not thread safe.
 *
 * @todo Look at thread safe implementation.
 */
void __kithip_mem_free(void *ptr);

/**
 * Free only the HIP portion of the given managed memory allocation. The
 * allocation referred to by `ptr` can be any HIP-based allocation and
 * *importantly* only this allocation component will be freed; the runtime's
 * data structures will not be updated. This is most often used by the runtime
 * during cleanup at application exit and in general is likely not what you want
 * to use in most other use cases.
 *
 * @param ptr A pointer to a HIP (managed or otherwise) memory allocation.
 */
void __kithip_mem_destroy(void *ptr);

/**
 * Request that the memory allocation associated with the given pointer be
 * prefetched to GPU memory. The memory must have been allcoated as managed
 * memory using the Kitsune runtime interface. If the provided pointer is not
 * recognized by the runtime this call is a silent no-op.
 *
 * @param ptr The pointer to the allocated region to prefetch.
 *
 * **NOTE**: See `__kithip_mem_host_prefetch()` for host-side prefetch requests.
 */
void *__kithip_mem_gpu_prefetch(void *ptr, void *opaque_stream);

/**
 * Request that the memory allocation associated with the given pointer be
 * prefetched to the host (CPU) memory. The memory must have been allocated as
 * managed memory using the Kitsune runtime interface. If the provided pointer
 * is not recognized by the runtime this call is a silent no-op.
 *
 * @param ptr The pointer to the allocated region to prefetch.
 *
 * **NOTE**: See `__kithip_mem_gpu_prefetch()` for GPU prefetch requests.
 */
void __kithip_mem_host_prefetch(void *ptr);

/**
 * Find symbol named \p sym_name in the module represented \p fat_bin.
 */
void *__kithip_get_global_symbol(void *fat_bin, const char *sym_name);

/**
 * Copy the given symbol from host memory to device memory. This is most often
 * used when doing code generation for global values that need to have identical
 * values on the device side (GPU). These calls are typically issued prior to
 * kernel launch to make sure values are synchronized.
 *
 * @param host_sym a pointer to the host-side symbol.
 * @param dev_sym handle to the device-side symbol.
 * @param bytes size in bytes of the symbol.
 */
void __kithip_memcpy_sym_to_device(void *host_sym, void *dev_sym, size_t bytes);

/**
 * Given a pointer to a fat binary, launch the named kernel with the given
 * arguments. The kernel will generally have been generated from a loop nest in
 * source code. The provided trip counts are for the loops from which the kernel
 * was derived. For the current Kitsune use cases the compiler will embed the
 * fat binary into the final executable and the arguments will be determined
 * during outlining.
 *
 * The runtime will determine an underlying target stream based on the calling
 * thread. It is currently assumed that the calling thread will issue any
 * prefetch, launch, and synchronization calls on its assigned stream. This
 * implementation path deprecates previous APIs that exposed streams as part of
 * the API.
 *
 * @param fatbin The fat binary image containing the compiled kernel.
 * @param name The name of the kernel to launch.
 * @param args The argument buffer for the kernel.
 * @param tc_z The trip count of the loop at depth 3, or 0 if the kernel being
 *             launched was derived from a loop nest with depth less than 3.
 * @param tc_y The trip count of the loop at depth 2, or 0 if the kernel being
 *             launched was derived from a loop nest with depth less than 2.
 * @param tc_x The trip count of the loop at depth 1 in the loop nest from which
 *             this kernel was derived.
 * @param tpb The number of threads per block to use. If this is 0, an
 *            appropriate value will be calculated.
 * @param inst_mix external static code analysis details.
 * @param stream_in externally created stream for execution.
 */
void *__kithip_launch_kernel(const void *fatbin, const char *name, void **args,
                             int64_t tc_z, int64_t tc_y, int64_t tc_x, int tpb,
                             const KitRTInstMix *inst_mix, void *stream_in);

/**
 * Set the runtime's value for the number of threads-per-block used in simple
 * launch parameter calculations.
 *
 * @param nthreads - number of threads per block
 */
void __kithip_set_threads_per_blk(int nthreads);

/**
 * Set the runtime's value for the maximum number of threads allowed per block
 * (typically a hardware limit). For HIP there is a dependency between the
 * runtime and the compiler-generated code (kernel) attributes. The compiler
 * will insert a call to this to match any specific code generation details that
 * were used.
 */
void __kithip_set_max_threads_per_blk(int nthreads);

/**
 * Provide a set of set of launch parameters to use for subsequent kernel launch
 * calls.  If this call is not made prior to a launch call a set of default
 * values will be used. Specifically a given number of threads will be set and
 * then a classic HIP (CUDA) style computation will be used to determine the
 * other details. At present the default threads-per-block is set to `256` and
 * may be controlled via the `KITHIP_THREADS_PER_BLOCK` environment variable at
 * runtime.
 *
 * **NOTE** These settings will remain in effect for all launch calls from the
 * calling point forward, i.e. it tweaks a global state. Therefore this call
 * is not thread safe as it is possible to race over these values.
 *
 * @param blks_per_grid The number of blocks in a grid for the subsequent
 *        launch call(s).
 * @param threads_per_blk The number of threads in a block for the subsequent
 *        launch call(s).
 *
 * @todo Make these parameters thread safe and bound to the next
 * kernel launch call(s).
 */
void __kithip_set_custom_launch_params(unsigned blks_per_grid,
                                       unsigned threads_per_blk);

/**
 * Return the stream that is assocaited with the calling thread. If a stream has
 * not yet been created and associated with the calling thread it will be
 * created and returned.
 */
void *__kithip_get_thread_stream(void);

/**
 * Synchronize the calling thread with its assocaited stream.
 */
void __kithip_sync_thread_stream(void *opaque_stream);

/**
 * Synchronize the host-side with **all** underlying streams in the current HIP
 * context. There a certain cases where we can't guarantee correctness without
 * blocking the CPU execution until all work is complete on the GPU. This is
 * obviously not always ideal in terms of performance but for correctness this
 * makes code generation sound when analysis is unavailable or inadequate.
 */
void __kithip_sync_context(void);

/**
 * Destroy the stream that is associated with the calling thread. If a stream
 * has not been assigned to the thread this call will simply return and function
 * as a no-op.
 */
void __kithip_delete_thread_stream(void *opaque_stream);

/**
 * Destroy all the thread-associated streams that are being managed by the
 * runtime. This call will essentially use HIP to destroy each stream and the
 * associated data structure entries used by the runtime.
 */
void __kithip_destroy_thread_streams(void);

/**
 * Has the HIP portion of the Kitsune runtime been successfully initialized?
 */
bool __kithip_is_initialized(void);

#ifdef __cplusplus
} // extern "C"
#endif

#define HIP_SAFE_CALL(x)                                                       \
  {                                                                            \
    hipError_t hip_result = x;                                                 \
    if (hip_result != hipSuccess) {                                            \
      kitrt::error("kithip", "%s:%d", __FILE__, __LINE__);                     \
      __kitrt_print_stack_trace();                                             \
      kitrt::error("kithip", "%s failed ('%s')", #x,                           \
                   hipGetErrorName(hip_result));                               \
      kitrt::fatal("kithip", hipGetErrorString(hip_result));                   \
    }                                                                          \
  }

#endif
