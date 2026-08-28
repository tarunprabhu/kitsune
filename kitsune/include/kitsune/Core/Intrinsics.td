//===- Intrinsics.td - Kitsune-specific intrinsics ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Kitsune-specific intrinsics.
//
//===----------------------------------------------------------------------===//

// ----------------------------------------------------------------------------
//
//                                  IMPORTANT
//
// - All intrinsic names *must* start with `kit_`.
//
// - Intrinsics that are asynchronous *must* start with `kit_async`. Other parts
//   of the code expect this. Failing to do so could result in miscompilation
//   since analyses may make incorrect assumptions about the available of the
//   data being operated on by these asynchronous intrinsics.
//
// - Intrinsics that are specific to the CPU-centric tapir targets must start
//   with `kit_cpu` or `kit_async_cpu`. Intrinsics that are specific to the
//   GPU-centric tapir targets must start with `kit_gpu` or `kit_async_gpu`.
//
// - The first argument of Kitsune's intrinsics must be a TTID. This must be a
//   32-bit integer and a compile-time constant, i.e. it must have the ImmArg
//   attribute.
//
// - Boolean arguments must have type i8, not i1. This is to simplify lowering
//   since, in the runtime, booleans are always 8-bit.
//
// - All intrinsics *must* be documented. We intentionally do not allow
//   undocumented intrinsics.
//
//----------------------------------------------------------------------------//

// The file is included in llvm/IR/Intrinsics.td. Specifying the full relative
// path this way works with all the ways that file is used in the repo.
include "../../kitsune/include/kitsune/Core/IntrinsicDocs.td"
include "../../kitsune/include/kitsune/Core/LibFuncs.td"

// Pointers to data in Kitsune's mobile address space.
def llvm_mobile_ptr_ty : LLVMQualPointerType<67>;

// A representaion of a tapir target. The definitions here should be kept in
// sync with the corresponding members of the llvm::TTID enum. The spelling of
// each def should match that of the llvm::TTID enum member.
class TTID;
def Cuda : TTID;
def Hip : TTID;
def OpenCilk : TTID;
def OpenMP : TTID;
def Pthreads : TTID;
def Qthreads : TTID;
def Serial : TTID;

// A pair of a tapir target and a runtime function. This is used in Kitsune's
// intrinsics when they support default lowering.
class RT<TTID TTParam, KitFunc FuncParam> {
  TTID TT = TTParam;
  KitFunc Func = FuncParam;
}

// Base class for all Kitsune-specific intrinsics. This simply extends the
// standard LLVM intrinsic class with some fields that allow documentation to
// be generated automatically.
class KitIntrinsic<list<LLVMType> ret_types,
                   list<LLVMType> param_types = [],
                   list<IntrinsicProperty> intr_properties = [],
                   string name = "",
                   list<SDNodeProperty> sd_properties = [],
                   bit disable_default_attributes = true>
    : Intrinsic<ret_types, param_types, intr_properties, name, sd_properties,
                disable_default_attributes> {
  // Documentation object for the intrinsic. This is required on all
  // Kitsune-specific intrinsics.
  KitDocIntrinsic Doc;

  // When using default lowering, the signatures of Kitsune's intrinsics and
  // the corresponding runtime functions are expected to match (since the first
  // argument of Kitsune's intrinsics will always be the TTID, the type of the
  // i'th argument of the corresponding runtime must be exactly the same as the
  // `i-1`th argument of the intrinsic). In some cases, however, there might be
  // a "benign" mismatch, where a simple cast is sufficient to resolve the
  // mismatch, and is also safe.
  //
  // For instance, the memcpy functions in Kitsune's cuda runtime expect a value
  // CUdeviceptr for the device-side address. On 64-bit systems - the only
  // architecture that we currently - CUdeviceptr actually maps to `uint64_t.`
  // Under the hood though, the actual value seems to be just a pointer cast to
  // an integer. Since the corresponding memcpy intrinsics operate on pointers
  // anyway, we can safely cast these arguments to integers when lowering to
  // cuda's memcpy intrinsics.
  //
  // If AllowParamCast is set to true, simple casts may be performed on any
  // argument during lowering to ensure that there is no type mismatch.
  //
  // NOTE: This must be used with caution. Incorrectly casting values may result
  // in hard to diagnose bugs.
  //
  // TODO: Having a simple flag here theoretically increases the possiblity of
  // introducing bugs by allowing casts of parameters where there is no reason
  // to allow casts. At this time, it is probably not worth the extra trouble
  // of turning this into a list.
  bit AllowParamCast = false;

  // Similar to allowing casts of the arguments, setting AllowReturnCast to
  // true will enable casting of the return type. This should only be used with
  // the mobile pointer intrinsics that return a mobile pointer. The runtime
  // functions will are unlikely to ever return a mobile pointer, so a cast will
  // be required.
  bit AllowReturnCast = false;

  // Some intrinsics require custom lowering. Some common situations when custom
  // lowering may be required include:
  //
  //   - When the signatures of the intrinsic and the runtime function to use
  //     when lowering do not match, and simple casting is not possible
  //
  //   - When simply replacing the intrinsic with a runtime function, even with
  //     casting is not sufficient. This can happen if the arguments need to be
  //     preprocessed, or additional memory must be used, for instance.
  //
  //   - When only the TTID alone is not sufficient to determine the runtime
  //     function to use. This is often the case with overloaded intrinsics.
  //     Kitsune's runtime functions strictly expose a C interface, so
  //     the runtime functions cannot be overloaded.
  bit RequiresCustomLowering = false;

  // If this intrinsic does not require custom lowering, this is a map from
  // a tapir target to the corresponding runtime function. If this map is not
  // present, it implies that custom lowering of the intrinsic is required.
  // If a tapir target is not present as an entry in the map, then that
  // intrinsic cannot be lowered with the given tapir target. Attempting to do
  // so will result in a catastrophic failure during compilation.
  list<RT> RTSpec = [];
}

//------------------------------------------------------------------------------
// Intrinsics for parallel execution on CPU's.

// Allocate memory on the heap.
def int_kit_cpu_malloc
    : KitIntrinsic<[llvm_ptr_ty], [llvm_i32_ty, llvm_i64_ty],
                   [NoAlias<RetIndex>, ImmArg<ArgIndex<0>>]> {
  let Doc = KitCPUMallocDoc;
  let RTSpec = [RT<OpenCilk, kitrt_malloc>,
                RT<OpenMP, kitrt_malloc>,
                RT<Pthreads, kitrt_malloc>,
                RT<Qthreads, kitrt_malloc>,
                RT<Serial, kitrt_malloc>];
}

// Free memory previously allocated with kit.cpu.malloc.
def int_kit_cpu_free
    : KitIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitCPUFreeDoc;
  let RTSpec = [RT<OpenCilk, kitrt_free>,
                RT<OpenMP, kitrt_free>,
                RT<Pthreads, kitrt_free>,
                RT<Qthreads, kitrt_free>,
                RT<Serial, kitrt_free>];
}

// Get the number of parallel CPU threads available.
def int_kit_cpu_num_threads
    : KitIntrinsic<[llvm_i64_ty], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitCPUNumThreadsDoc;
  let RTSpec = [RT<OpenCilk, kitocilk_num_workers>,
                RT<OpenMP, kitomp_num_threads>,
                RT<Pthreads, kitpthr_num_threads>,
                RT<Qthreads, kitqthr_num_workers>];
}

// Get the ID of the thread from which this is called.
def int_kit_cpu_thread_id
    : KitIntrinsic<[llvm_i64_ty], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitCPUThreadIDDoc;
  let RTSpec = [RT<OpenCilk, kitocilk_worker_id>,
                RT<OpenMP, kitomp_thread_id>,
                RT<Pthreads, kitpthr_thread_id>,
                RT<Qthreads, kitqthr_worker_id>];
}

// Launch a function on one or more CPU threads (asynchronous).
def int_kit_async_cpu_threads_launch
    : KitIntrinsic<[llvm_ptr_ty],
                   [llvm_i32_ty,  // TTID
                    llvm_ptr_ty,  // thread function
                    llvm_i64_ty,  // start index
                    llvm_i64_ty,  // end index
                    llvm_vararg_ty],
                   [NonNull<RetIndex>, ImmArg<ArgIndex<0>>,
                    NonNull<ArgIndex<1>>]> {
  let Doc = KitAsyncLaunchThreadsDoc;
  let RTSpec = [RT<Pthreads, kitpthr_async_launch>];
  let RequiresCustomLowering = true;
}

// Launch a function on one or more CPU threads (synchronous).
def int_kit_cpu_threads_launch
    : KitIntrinsic<[],
                   [llvm_i32_ty,  // TTID
                    llvm_ptr_ty,  // thread function
                    llvm_i64_ty,  // start index
                    llvm_i64_ty,  // end index
                    llvm_vararg_ty],
                   [ImmArg<ArgIndex<0>>, NonNull<ArgIndex<1>>]> {
  let Doc = KitLaunchThreadsDoc;
  let RTSpec = [RT<OpenMP, kitomp_launch>,
                RT<Qthreads, kitqthr_launch>];
  let RequiresCustomLowering = true;
}

// Sync the threads launched by an earlier call to
// llvm.kit.async.cpu.threads.launch.
def int_kit_cpu_threads_sync
    : KitIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty],
                   [ImmArg<ArgIndex<0>>, NonNull<ArgIndex<1>>]> {
  let Doc = KitSyncThreadsDoc;
  let RTSpec = [RT<Pthreads, kitpthr_sync>];
}

//------------------------------------------------------------------------------
// Intrinsics for parallel execution on GPU's.

// Get the thread index in the x dimension in a GPU kernel.
def int_kit_gpu_thread_id_x
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUTIDXDoc;
}

// Get the thread index in the y dimension in a GPU kernel.
def int_kit_gpu_thread_id_y
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUTIDYDoc;
}

// Get the thread index in the z dimension in a GPU kernel.
def int_kit_gpu_thread_id_z
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUTIDZDoc;
}

// Get the block index in the x dimension in a GPU kernel.
def int_kit_gpu_block_id_x
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUBlockIDXDoc;
}

// Get the block index in the y dimension in a GPU kernel.
def int_kit_gpu_block_id_y
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUBlockIDYDoc;
}

// Get the block index in the z dimension in a GPU kernel.
def int_kit_gpu_block_id_z
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUBlockIDZDoc;
}

// Get the number of blocks in the x dimension in a GPU kernel.
def int_kit_gpu_block_size_x
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUBlockSizeXDoc;
}

// Get the number of blocks in the y dimension in a GPU kernel.
def int_kit_gpu_block_size_y
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUBlockSizeYDoc;
}

// Get the number of blocks in the z dimension in a GPU kernel.
def int_kit_gpu_block_size_z
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUBlockSizeZDoc;
}

// Get the number of grids in the x dimension in a GPU kernel.
def int_kit_gpu_grid_size_x
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUGridSizeXDoc;
}

// Get the number of blocks in the y dimension in a GPU kernel.
def int_kit_gpu_grid_size_y
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUGridSizeYDoc;
}

// Get the number of blocks in the z dimension in a GPU kernel.
def int_kit_gpu_grid_size_z
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUGridSizeZDoc;
}

// Get the number of threads per warp.
// NOTE: This is not really convergent. Doing so is overly conservative, but,
// this way, it does not get hoisted out of tapir loops. Once we fix the
// lowering of warp shuffles, we can probably remove this attribute.
def int_kit_gpu_warp_size
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                   [IntrConvergent, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUWarpSizeDoc;
}

// Get the ID of a warp within a block in a GPU kernel.
def int_kit_gpu_warp_id
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                    IntrConvergent, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUWarpIDDoc;
}

// Get the lane number of a thread within a warp in a GPU kernel.
def int_kit_gpu_warp_lane
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                    IntrConvergent, IntrNoMem, IntrWillReturn]> {
  let Doc = KitGPUWarpLaneDoc;
}

// The equivalent of the __shfl_down_sync builtin available in cuda and hip.
def int_kit_gpu_warp_shfl_down_sync
    : KitIntrinsic<[llvm_any_ty], [llvm_i32_ty, llvm_any_ty, llvm_i32_ty],
                   [ImmArg<ArgIndex<0>>, IntrConvergent, IntrWillReturn]> {
  let Doc = KitGPUWarpShflDownSyncDoc;
}

// Perform a simple reduction on a GPU where threads accumulate their
// contribution to the final result independently of all other threads.
def int_kit_gpu_reduce_direct
    : KitIntrinsic<[],
                   [llvm_i32_ty,     // TTID
                    llvm_i32_ty,     // The reduction operator enum value
                    llvm_ptr_ty,     // Pointer to the reduced result
                    llvm_i32_ty,     // Size, in bytes, of the reduced result
                    llvm_any_ty,     // The value to pass to the reducer
                    llvm_any_ty,     // The unit value for the reducer
                    llvm_ptr_ty,     // The reducer function
                    llvm_vararg_ty], // Additional reducer arguments
                   [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                    NonNull<ArgIndex<2>>, ImmArg<ArgIndex<3>>,
                    NonNull<ArgIndex<6>>, ReadOnly<ArgIndex<6>>,
                    IntrConvergent, IntrNoFree, IntrWillReturn]> {
  let Doc = KitGPUReduceDirectDoc;
}

// Perform a reduction on a GPU using shared memory.
def int_kit_gpu_reduce_shared_memory
    : KitIntrinsic<[],
                   [llvm_i32_ty,     // TTID
                    llvm_i32_ty,     // The reduction operator enum value
                    llvm_ptr_ty,     // Pointer to the reduced result
                    llvm_i32_ty,     // Size, in bytes, of the reduced result
                    llvm_any_ty,     // The value to pass to the reducer
                    llvm_any_ty,     // The unit value for the reducer
                    llvm_ptr_ty,     // The reducer function
                    llvm_vararg_ty], // Additional reducer arguments
                   [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                    NonNull<ArgIndex<2>>, ImmArg<ArgIndex<3>>,
                    NonNull<ArgIndex<6>>, ReadOnly<ArgIndex<6>>,
                    IntrConvergent, IntrNoFree, IntrWillReturn]> {
  let Doc = KitGPUReduceSharedMemoryDoc;
}

// Perform a reduction on a GPU using the warp shuffle technique without any
// shared memory.
def int_kit_gpu_reduce_warp_shuffle
    : KitIntrinsic<[],
                   [llvm_i32_ty,     // TTID
                    llvm_i32_ty,     // The reduction operator enum value
                    llvm_ptr_ty,     // Pointer to the reduced result
                    llvm_i32_ty,     // Size, in bytes, of the reduced result
                    llvm_any_ty,     // The value to pass to the reducer
                    llvm_any_ty,     // The unit value for the reducer
                    llvm_ptr_ty,     // The reducer function
                    llvm_vararg_ty], // Additional reducer arguments
                   [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                    NonNull<ArgIndex<2>>, ImmArg<ArgIndex<3>>,
                    NonNull<ArgIndex<6>>, ReadOnly<ArgIndex<6>>,
                    IntrConvergent, IntrNoFree, IntrWillReturn]> {
  let Doc = KitGPUReduceWarpShuffleDoc;
}

// Perform a reduction on a GPU using the warp shuffle technique together with
// shared memory
def int_kit_gpu_reduce_warp_shuffle_shared_memory
    : KitIntrinsic<[],
                   [llvm_i32_ty,     // TTID
                    llvm_i32_ty,     // The reduction operator enum value
                    llvm_ptr_ty,     // Pointer to the reduced result
                    llvm_i32_ty,     // Size, in bytes, of the reduced result
                    llvm_any_ty,     // The value to pass to the reducer
                    llvm_any_ty,     // The unit value for the reducer
                    llvm_ptr_ty,     // The reducer function
                    llvm_vararg_ty], // Additional reducer arguments
                   [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                    NonNull<ArgIndex<2>>, ImmArg<ArgIndex<3>>,
                    NonNull<ArgIndex<6>>, ReadOnly<ArgIndex<6>>,
                    IntrConvergent, IntrNoFree, IntrWillReturn]> {
  let Doc = KitGPUReduceWarpShuffleSharedMemoryDoc;
}

// Get the number of compute units available to a GPU tapir target.
def int_kit_gpu_num_compute_units
    : KitIntrinsic<[llvm_i64_ty], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPUNumComputeUnitsDoc;
  let RTSpec = [RT<Cuda, kitcuda_num_sms>,
                RT<Hip, kithip_num_cus>];
}

// Allocate memory on the GPU.
def int_kit_gpu_malloc
    : KitIntrinsic<[llvm_ptr_ty], [llvm_i32_ty, llvm_i64_ty],
                   [NoAlias<RetIndex>, ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPUMallocDoc;
  let RTSpec = [RT<Cuda, kitcuda_malloc>,
                RT<Hip, kithip_malloc>];
}

// Free memory previously allocated with kit.gpu.malloc.
def int_kit_gpu_free
    : KitIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPUFreeDoc;
  let RTSpec = [RT<Cuda, kitcuda_free>,
                RT<Hip, kithip_free>];
}

// Get a new GPU stream.
def int_kit_gpu_stream_new
    : KitIntrinsic<[llvm_ptr_ty], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPUGetStreamDoc;
  let RTSpec = [RT<Cuda, kitcuda_stream_new>,
                RT<Hip, kithip_stream_new>];
}

// Synchronize a GPU stream.
def int_kit_gpu_stream_sync
    : KitIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPUSyncStreamDoc;
  let RTSpec = [RT<Cuda, kitcuda_stream_sync>,
                RT<Hip, kithip_stream_sync>];
}

// Launch a kernel on the GPU on the given stream.
def int_kit_async_gpu_kernel_launch
    : KitIntrinsic<[llvm_ptr_ty], // the stream (may be a new stream)
                   [llvm_i32_ty,  // TTID
                    llvm_ptr_ty,  // device code global
                    llvm_ptr_ty,  // name
                    llvm_i64_ty,  // tcZ, or 0
                    llvm_i64_ty,  // tcY, or 0
                    llvm_i64_ty,  // tcX
                    llvm_i32_ty,  // threads per block
                    llvm_ptr_ty,  // kernel properties
                    llvm_ptr_ty,  // stream
                    llvm_vararg_ty],
                   [ImmArg<ArgIndex<0>>]> {
  let Doc = KitAsyncGPULaunchKernelDoc;
  let RTSpec = [RT<Cuda, kitcuda_kernel_launch>,
                RT<Hip, kithip_kernel_launch>];
  let RequiresCustomLowering = true;
}

// Copy data from a device (usually a GPU) to the host.
def int_kit_async_gpu_memcpy_dtoh
    : KitIntrinsic<[],
                   [llvm_i32_ty,  // TTID
                    llvm_ptr_ty,  // pointer to host memory (destination)
                    llvm_ptr_ty,  // pointer to device memory (source)
                    llvm_i64_ty,  // number of bytes to copy
                    llvm_ptr_ty], // the stream on which to perform the copy
                   [ImmArg<ArgIndex<0>>]> {
  let Doc = KitAsyncMemcpyDToHDoc;
  let AllowParamCast = true;
}

// Copy data from the host to a device (usually a GPU).
def int_kit_async_gpu_memcpy_htod
    : KitIntrinsic<[],
                   [llvm_i32_ty,  // TTID
                    llvm_ptr_ty,  // pointer to device memory (destination)
                    llvm_ptr_ty,  // pointer to host memory (source)
                    llvm_i64_ty,  // number of bytes to copy
                    llvm_ptr_ty], // the stream on which to perform the copy
                   [ImmArg<ArgIndex<0>>]> {
  let Doc = KitAsyncMemcpyHToDDoc;
  let AllowParamCast = true;
}

// Copy bytes from the GPU to the host. This is a blocking call.
def int_kit_gpu_memcpy_dtoh
    : KitIntrinsic<[],
                   [llvm_i32_ty,  // TTID
                    llvm_ptr_ty,  // pointer to host memory (destination)
                    llvm_ptr_ty,  // pointer to device memory (source)
                    llvm_i64_ty], // number of bytes to copy
                   [ImmArg<ArgIndex<0>>]> {
  let Doc = KitMemcpyDToHDoc;
  let AllowParamCast = true;
  let RTSpec = [RT<Cuda, kitcuda_memcpy_dtoh>,
                RT<Hip, kithip_memcpy_dtoh>];
}

// Copy bytes from the host to the GPU. This is a blocking call.
def int_kit_gpu_memcpy_htod
    : KitIntrinsic<[],
                   [llvm_i32_ty,  // TTID
                    llvm_ptr_ty,  // pointer to device memory (destination)
                    llvm_ptr_ty,  // pointer to host memory (source)
                    llvm_i64_ty], // number of bytes to copy
                   [ImmArg<ArgIndex<0>>]> {
  let Doc = KitMemcpyHToDDoc;
  let AllowParamCast = true;
  let RTSpec = [RT<Cuda, kitcuda_memcpy_htod>,
                RT<Hip, kithip_memcpy_htod>];
}

// Initialize GPU memory.
def int_kit_gpu_memset
    : KitIntrinsic<[],
                   [llvm_i32_ty,     // TTID
                    llvm_ptr_ty,     // Pointer to device memory,
                    llvm_i64_ty,     // Number of bytes to initialize
                    llvm_any_ty,     // The initial value
                    llvm_vararg_ty],
                   [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPUMemsetDoc;
  let AllowParamCast = true;
  let RequiresCustomLowering = true;
}

// Issue a prefetch from a device (usually a GPU) to the host.
def int_kit_async_gpu_prefetch_dtoh
    : KitIntrinsic<[llvm_ptr_ty], // stream
                   [llvm_i32_ty,  // TTID
                    llvm_ptr_ty,  // pointer to memory to prefetch
                    llvm_i64_ty,  // number of bytes to prefetch
                    llvm_ptr_ty], // stream
                   [ImmArg<ArgIndex<0>>]> {
  let Doc = KitAsyncPrefetchDToHDoc;
  let RTSpec = [RT<Cuda, kitcuda_prefetch_dtoh>,
                RT<Hip, kithip_prefetch_dtoh>];
}

// Issue a prefetch from the host to a device (usually a GPU).
def int_kit_async_gpu_prefetch_htod
    : KitIntrinsic<[llvm_ptr_ty], // stream
                   [llvm_i32_ty,  // TTID
                    llvm_ptr_ty,  // pointer to memory to prefetch
                    llvm_i64_ty,  // number of bytes to prefetch
                    llvm_ptr_ty], // stream
                   [ImmArg<ArgIndex<0>>, IntrWillReturn]> {
  let Doc = KitAsyncPrefetchHToDDoc;
  let RTSpec = [RT<Cuda, kitcuda_prefetch_htod>,
                RT<Hip, kithip_prefetch_htod>];
}

// Register device code with the runtime.
def int_kit_gpu_register_devcode
    : KitIntrinsic<[llvm_ptr_ty], // A handle to be used for other registrations
                   [llvm_i32_ty,  // TTID
                    llvm_ptr_ty], // pointer to the device code
                   [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPURegisterDevCodeDoc;
  let RTSpec = [RT<Cuda, kitcuda_register_devcode>,
                RT<Hip, kithip_register_devcode>];
}

// Signal that all global variables in device code have been registered. This
// is only needed by the cuda runtime.
def int_kit_gpu_register_devcode_end
    : KitIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPURegisterDevCodeEndDoc;
  let RTSpec = [RT<Cuda, kitcuda_register_devcode_end>];
}

// Register a "normal" (non-UVM) global variable used in device code.
def int_kit_gpu_register_global
    : KitIntrinsic<[],
                   [llvm_i32_ty,  // TTID
                    llvm_ptr_ty,  // Handle returned by gpu.register.device.code
                    llvm_ptr_ty,  // Out pointer to get host address of global
                    llvm_ptr_ty,  // Name of the global in host code
                    llvm_ptr_ty,  // Name of the global in device code
                    llvm_i64_ty,  // Size of the global
                    llvm_i32_ty,  // Is the global external
                    llvm_i32_ty], // Is the global constant
                   [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<5>>,
                    ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>,
                    NonNull<ArgIndex<1>>, NonNull<ArgIndex<2>>,
                    NonNull<ArgIndex<3>>, NonNull<ArgIndex<4>>]> {
  let Doc = KitGPURegisterGlobalDoc;
  let RTSpec = [RT<Cuda, kitcuda_register_global>,
                RT<Hip, kithip_register_global>];
}

// Register a global variable, used in device code, that is to be allocated in
// managed memory (UVM).
def int_kit_gpu_register_global_managed
    : KitIntrinsic<[],
                   [llvm_i32_ty,  // TTID
                    llvm_ptr_ty,  // Handle returned by gpu.register.device.code
                    llvm_ptr_ty,  // Out pointer for UVM allocated address
                    llvm_ptr_ty,  // Address of global on the device
                    llvm_ptr_ty,  // Name of the global
                    llvm_i64_ty,  // Size of the global
                    llvm_i32_ty,  // Alignment of the global
                    llvm_i32_ty,  // Is the global external
                    llvm_i32_ty], // Is the global constant
                   [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<5>>,
                    ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>,
                    ImmArg<ArgIndex<8>>, NonNull<ArgIndex<1>>,
                    NonNull<ArgIndex<2>>, NonNull<ArgIndex<3>>,
                    NonNull<ArgIndex<4>>]> {
  let Doc = KitGPURegisterGlobalManagedDoc;
  let RTSpec = [RT<Cuda, kitcuda_register_global_managed>,
                RT<Hip, kithip_register_global_managed>];
}

// Get the address of a symbol on a GPU.
def int_kit_gpu_symbol_address
    : KitIntrinsic<[llvm_ptr_ty],
                   [llvm_i32_ty, llvm_ptr_ty, llvm_ptr_ty],
                   [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPUSymbolAddressDoc;
  let RTSpec = [RT<Cuda, kitcuda_symbol_address>,
                RT<Hip, kithip_symbol_address>];
}

// Unregister device code that was previously registered with
// int_kit_gpu_register_devcode.
def int_kit_gpu_unregister_devcode
    : KitIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPUUnregisterDevCodeDoc;
  let RTSpec = [RT<Cuda, kitcuda_unregister_devcode>,
                RT<Hip, kithip_unregister_devcode>];
}

//------------------------------------------------------------------------------
// Intrinsics for mobile buffers

// Allocate a mobile buffer.
def int_kit_mobile_alloc
    : KitIntrinsic<[llvm_mobile_ptr_ty], [llvm_i32_ty, llvm_i64_ty],
                   [NoAlias<RetIndex>, ImmArg<ArgIndex<0>>]> {
  let Doc = KitMobileAllocDoc;
  let AllowReturnCast = true;
  let RTSpec = [RT<Cuda, kitcuda_managed_malloc>,
                RT<Hip, kithip_managed_malloc>,
                RT<OpenCilk, kitrt_mem_alloc>,
                RT<OpenMP, kitrt_mem_alloc>,
                RT<Pthreads, kitrt_mem_alloc>,
                RT<Qthreads, kitrt_mem_alloc>,
                RT<Serial, kitrt_mem_alloc>];
}

// Free a mobile buffer previously allocated by llvm.kit.mobile.alloc.
def int_kit_mobile_free
    : KitIntrinsic<[], [llvm_i32_ty, llvm_mobile_ptr_ty],
                   [ImmArg<ArgIndex<0>>, NoCapture<ArgIndex<1>>,
                    ReadNone<ArgIndex<1>>]> {
  let Doc = KitMobileFreeDoc;
  let AllowParamCast = true;
  let RTSpec = [RT<Cuda, kitcuda_managed_free>,
                RT<Hip, kithip_managed_free>,
                RT<OpenCilk, kitrt_mem_free>,
                RT<OpenMP, kitrt_mem_free>,
                RT<Pthreads, kitrt_mem_free>,
                RT<Qthreads, kitrt_mem_free>,
                RT<Serial, kitrt_mem_free>];
}

// Initialize the elements of a contiguous, mobile buffer with the given value.
def int_kit_mobile_init
    : KitIntrinsic<[],
                   [llvm_i32_ty,        // TTID
                    llvm_mobile_ptr_ty, // pointer to memory to initialize
                    llvm_i64_ty,        // number of elements in the buffer
                    llvm_any_ty,        // the initializer
                    llvm_vararg_ty],
                   [ImmArg<ArgIndex<0>>]> {
  let Doc = KitMobileInitDoc;
  let AllowParamCast = true;
  let RequiresCustomLowering = true;
}

//------------------------------------------------------------------------------
// Intrinsics to support parallel reductions

// Reduce a single value using a given reduction function.
def int_kit_reduce_0
    : KitIntrinsic<[],
                   [llvm_i32_ty,     // TTID
                    llvm_i32_ty,     // The reduction operator enum value
                    llvm_ptr_ty,     // Pointer to result
                    llvm_i32_ty,     // Size, in bytes, of the reduced result
                    llvm_any_ty,     // The value to pass to the reducer
                    llvm_any_ty,     // The unit value for the reducer
                    llvm_ptr_ty,     // The reducer function
                    llvm_vararg_ty], // Additional reducer arguments
                   [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                    NonNull<ArgIndex<2>>, ImmArg<ArgIndex<3>>,
                    NonNull<ArgIndex<6>>, ReadOnly<ArgIndex<6>>]> {
  let Doc = KitReduce0Doc;
}
