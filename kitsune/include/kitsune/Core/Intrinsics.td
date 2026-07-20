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

// Pointers to data in Kitsune's mobile address space.
def llvm_mobile_ptr_ty : LLVMQualPointerType<67>;

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
}

//------------------------------------------------------------------------------
// Intrinsics for parallel execution on CPU's.

// Get the number of parallel CPU threads available.
def int_kit_cpu_num_threads
    : KitIntrinsic<[llvm_i64_ty], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitCPUNumThreadsDoc;
}

// Get the ID of the thread from which this is called.
def int_kit_cpu_thread_id
    : KitIntrinsic<[llvm_i64_ty], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitCPUThreadIDDoc;
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
}

// Sync the threads launched by an earlier call to
// llvm.kit.async.cpu.threads.launch.
def int_kit_cpu_threads_sync
    : KitIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty],
                   [ImmArg<ArgIndex<0>>, NonNull<ArgIndex<1>>]> {
  let Doc = KitSyncThreadsDoc;
}

//------------------------------------------------------------------------------
// Intrinsics for parallel execution on GPU's.

// Get the thread index in the x dimension in a GPU kernel.
def int_kit_gpu_thread_id_x
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let Doc = KitGPUTIDXDoc;
}

// Get the thread index in the y dimension in a GPU kernel.
def int_kit_gpu_thread_id_y
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let Doc = KitGPUTIDYDoc;
}

// Get the thread index in the z dimension in a GPU kernel.
def int_kit_gpu_thread_id_z
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let Doc = KitGPUTIDZDoc;
}

// Get the block index in the x dimension in a GPU kernel.
def int_kit_gpu_block_id_x
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let Doc = KitGPUBlockIDXDoc;
}

// Get the block index in the y dimension in a GPU kernel.
def int_kit_gpu_block_id_y
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let Doc = KitGPUBlockIDYDoc;
}

// Get the block index in the z dimension in a GPU kernel.
def int_kit_gpu_block_id_z
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let Doc = KitGPUBlockIDZDoc;
}

// Get the number of blocks in the x dimension in a GPU kernel.
def int_kit_gpu_block_size_x
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let Doc = KitGPUBlockSizeXDoc;
}

// Get the number of blocks in the y dimension in a GPU kernel.
def int_kit_gpu_block_size_y
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let Doc = KitGPUBlockSizeYDoc;
}

// Get the number of blocks in the x dimension in a GPU kernel.
def int_kit_gpu_block_size_z
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let Doc = KitGPUBlockSizeZDoc;
}

// Get the number of grids in the x dimension in a GPU kernel.
def int_kit_gpu_grid_size_x
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let Doc = KitGPUGridSizeXDoc;
}

// Get the number of blocks in the y dimension in a GPU kernel.
def int_kit_gpu_grid_size_y
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let Doc = KitGPUGridSizeYDoc;
}

// Get the number of blocks in the x dimension in a GPU kernel.
def int_kit_gpu_grid_size_z
    : KitIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let Doc = KitGPUGridSizeZDoc;
}

// Get a new GPU stream.
def int_kit_gpu_stream_new
    : KitIntrinsic<[llvm_ptr_ty], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPUGetStreamDoc;
}

// Synchronize a GPU stream.
def int_kit_gpu_stream_sync
    : KitIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPUSyncStreamDoc;
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
}

// Get the address of a symbol on a GPU.
def int_kit_gpu_symbol_address
    : KitIntrinsic<[llvm_ptr_ty],
                   [llvm_i32_ty, llvm_ptr_ty, llvm_ptr_ty],
                   [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPUSymbolAddressDoc;
}

// Copy a symbol from device to host.
// FIXME: The runtime call here simply calls cuMemcpyDtoH. We should consider
// removing this and using kit_memcpy_dtoh instead.
def int_kit_gpu_symbol_memcpy_dtoh
    : Intrinsic<[],
                [llvm_i32_ty,  // TTID
                 llvm_ptr_ty,  // pointer to symbol on the host
                 llvm_ptr_ty,  // pointer to symbol on the device
                 llvm_i64_ty], // number of bytes to copy
                [ImmArg<ArgIndex<0>>]>;

// Copy a symbol from host to device.
// FIXME: The runtime call here simply calls cuMemcpyHtoD. We should consider
// removing this and using kit_memcpy_htod instead.
def int_kit_gpu_symbol_memcpy_htod
    : Intrinsic<[],
                [llvm_i32_ty,  // TTID
                 llvm_ptr_ty,  // pointer to symbol on the device
                 llvm_ptr_ty,  // pointer to symbol on the host
                 llvm_i64_ty], // number of bytes to copy
                [ImmArg<ArgIndex<0>>]>;

// Register device code with the runtime.
def int_kit_gpu_register_devcode
    : KitIntrinsic<[llvm_ptr_ty], // A handle to be used for other registrations
                   [llvm_i32_ty,  // TTID
                    llvm_ptr_ty], // pointer to the device code
                   [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPURegisterDevCodeDoc;
}

// Signal that all global variables in device code have been registered. This
// is only needed by the cuda runtime.
def int_kit_gpu_register_devcode_end
    : KitIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPURegisterDevCodeEndDoc;
}

// Unregister device code that was previously registered with
// int_kit_gpu_register_devcode.
def int_kit_gpu_unregister_devcode
    : KitIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitGPUUnregisterDevCodeDoc;
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
}

//------------------------------------------------------------------------------
// Intrinsics for mobile buffers

// Allocate a mobile buffer.
def int_kit_mobile_alloc
    : KitIntrinsic<[llvm_mobile_ptr_ty], [llvm_i32_ty, llvm_i64_ty],
                   [NoAlias<RetIndex>, ImmArg<ArgIndex<0>>]> {
  let Doc = KitMobileAllocDoc;
}

// Free a mobile buffer previously allocated by llvm.kit.mobile.alloc.
def int_kit_mobile_free
    : KitIntrinsic<[], [llvm_i32_ty, llvm_mobile_ptr_ty],
                   [ImmArg<ArgIndex<0>>, NoCapture<ArgIndex<1>>,
                    ReadNone<ArgIndex<1>>]> {
  let Doc = KitMobileFreeDoc;
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
}

//------------------------------------------------------------------------------
// Intrinsics to support parallel reductions

// Reduce a single value using a given reduction function.
def int_kit_reduce_0
    : KitIntrinsic<[],
                   [llvm_i32_ty,     // TTID
                    llvm_ptr_ty,     // Pointer to result
                    llvm_i32_ty,     // sizeof(result)
                    llvm_any_ty,     // The value to pass to the reducer
                    llvm_any_ty,     // The unit value for the reducer
                    llvm_ptr_ty,     // The reducer function
                    llvm_vararg_ty], // Additional reducer arguments
                   [ImmArg<ArgIndex<0>>, NonNull<ArgIndex<1>>,
                    ImmArg<ArgIndex<2>>, NonNull<ArgIndex<5>>,
                    ReadOnly<ArgIndex<5>>]> {
  let Doc = KitReduce0Doc;
}

// Reduce the values in a contiguous 1D array using a given reduction function.
def int_kit_reduce_1
    : KitIntrinsic<[],
                   [llvm_i32_ty,        // TTID
                    llvm_ptr_ty,        // Pointer to result
                    llvm_i32_ty,        // sizeof(result)
                    llvm_mobile_ptr_ty, // Pointer to 1D array to reduce
                    llvm_i64_ty,        // The number of elements in the array
                    llvm_any_ty,        // The unit value for the reducer
                    llvm_ptr_ty,        // The reducer function
                    llvm_vararg_ty],    // Additional reducer arguments
                   [ImmArg<ArgIndex<0>>, NonNull<ArgIndex<1>>,
                    ImmArg<ArgIndex<2>>, ReadOnly<ArgIndex<3>>,
                    NonNull<ArgIndex<6>>, ReadOnly<ArgIndex<6>>]> {
  let Doc = KitReduce1DDoc;
}

// Get the number of partial reductions to perform when lowering a parallel
// reduction.
def int_kit_reduce_num_partials
    : KitIntrinsic<[llvm_i64_ty],
                   [llvm_i32_ty,  // TTID
                    llvm_i64_ty], // Reduction loop trip count
                   [ImmArg<ArgIndex<0>>]> {
  let Doc = KitReduceNumPartialsDoc;
}

//------------------------------------------------------------------------------
// Intrinsics to control Kitsune's runtime(s).

// Finalize Kitsune's runtime.
def int_kit_runtime_finalize
    : KitIntrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitFinalizeDoc;
}

// Initialize Kitsune's runtime.
def int_kit_runtime_initialize
    : KitIntrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]> {
  let Doc = KitInitializeDoc;
}

// Enable xnack in the runtime. This is only relevant for the hip tapir target
// and its associated kitsune runtime.
def int_kit_runtime_set_xnack
    : KitIntrinsic<[], [llvm_i32_ty, llvm_i8_ty], []> {
  let Doc = KitGPUEnableXnackDoc;
}

// Enable the y-axis launch pattern for all GPU kernels.
def int_kit_runtime_set_y_axis_kernel_launch
    : KitIntrinsic<[], [llvm_i32_ty, llvm_i8_ty],
                   [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]> {
  let Doc = KitGPUEnableYAxisKernelLaunchDoc;
}
