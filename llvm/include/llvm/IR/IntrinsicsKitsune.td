//===- IntrinsicsKitsune.td - Defines Kitsune intrinsics ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Kitsune-specific intrinsics. In the intrinsics here,
// boolean arguments have type i8 instead of i1 to simplify the lowering since,
// in the runtime, booleans are always 8-bit.
//
//===----------------------------------------------------------------------===//

def llvm_mobile_ptr_ty : LLVMQualPointerType<67>; // Kitsune mobile pointers

//===----------------------------------------------------------------------===//

// Mobile memory allocator. This will allocate the given number of bytes as a
// mobile buffer. The lowering of this intrinsic will depend on
//
//   - The Tapir target(s) that are enabled
//   - The uses of the buffer
//
// TODO: This should eventually return a pointer in a specific address space.
// This may require the returned type to be replaced with an
// LLVMQualPointerType.
def int_kitsune_mobile_alloc
    : Intrinsic<[llvm_mobile_ptr_ty], [llvm_i64_ty],
                [IntrInaccessibleMemOnly, NoAlias<RetIndex>]>;

// Mobile memory deallocator. This will free memory that was previously
// allocated by kitsune_mobile_alloc.
def int_kitsune_mobile_free
    : Intrinsic<[], [llvm_mobile_ptr_ty],
                [IntrInaccessibleMemOrArgMemOnly,
                 NoCapture<ArgIndex<0>>, ReadNone<ArgIndex<0>>]>;

// Enable launch refinement. Setting this will cause the runtime to calculate
// a kernel's launch parameters based on some heuristic. In the case of NVIDIA
// GPU's, this is an occupancy-based heuristic provided by cuda's API.
//
// NOTE: As of May 2025, this is only implemented in kitsune's cuda runtime. If
// this is implemented for kitsune's hip runtime as well, this comment should be
// updated.
def int_kitrt_enable_refine_launches
    : Intrinsic<[],
                [llvm_i8_ty,  // tapir target id
                 llvm_i8_ty], // Non-zero to refine launches, zero otherwise
                [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                 IntrInaccessibleMemOnly, IntrHasSideEffects]>;

// Call a function in kitsune's runtime that enables verbose mode.
def int_kitrt_enable_verbose
    : Intrinsic<[],
                [llvm_i8_ty], // Non-zero to enable verbose mode, zero otherwise
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Enable xnack in the runtime. This is only relevant for the hip tapir target
// and its associated kitsune runtime.
def int_kitrt_hip_enable_xnack
    : Intrinsic<[],
                [llvm_i8_ty], // Non-zero to enable xnack, zero otherwise
                [IntrInaccessibleMemOnly, IntrHasSideEffects]>;

// Enable the y-axis launch pattern for all GPU kernels.
//
// NOTE: As of May 2025, this is only implemented in kitsune's hip runtime. If
// this is implemented for kitsune's cuda runtime as well, this comment should
// be updated.
def int_kitrt_enable_y_axis_launches
    : Intrinsic<[],
                [llvm_i8_ty,  // tapir target id
                 llvm_i8_ty], // Non-zero to enable y-axis launches, 0 otherwise
                [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                 IntrInaccessibleMemOnly, IntrHasSideEffects]>;

// Call the appropriate finalization function from the kitsune runtime.
def int_kitrt_finalize
    : Intrinsic<[],
                [llvm_i8_ty], // tapir target id
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOnly,
                 IntrHasSideEffects]>;

// Initialize the appropriate kitsune runtime. There are separate initialization
// functions in kitrt for the runtime components of certain tapir targets.
def int_kitrt_initialize
    : Intrinsic<[],
                [llvm_i8_ty], // tapir target id
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOnly,
                 IntrHasSideEffects]>;

// Launch a kernel on the GPU on the given stream. Returns the same stream. If
// a stream is not provided, one will be created and returned. The stream is a
// pointer to an opaque object.
def int_kitrt_launch_kernel
    : Intrinsic<[llvm_ptr_ty], // stream
                [llvm_i8_ty,   // tapir target id
                 llvm_ptr_ty,  // fat binary
                 llvm_ptr_ty,  // name
                 llvm_ptr_ty,  // args[]
                 llvm_i64_ty,  // trip_count
                 llvm_i32_ty,  // threads per block
                 llvm_ptr_ty,  // instruction mix info
                 llvm_ptr_ty], // stream
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Issue a prefetch from the host to the GPU. This will be lowered to an
// appropriate call in Kitsune's runtime. The function takes a pointer to
// an opaque stream on which to issue the prefetch, or null to indicate that a
// new stream should be created. Returns the pointer to the stream - if one was
// given - or a new stream that was created. A pointer to the buffer to prefetch
// is also provided. The number of bytes to prefetch may be -1 which indicates
// that the runtime is required to determine the number of bytes to prefetch.
def int_kitrt_prefetch_device
    : Intrinsic<[llvm_ptr_ty], // stream
                [llvm_i8_ty,   // tapir target id
                 llvm_ptr_ty,  // pointer to memory to prefetch
                 llvm_i64_ty,  // number of bytes to prefetch
                 llvm_ptr_ty], // stream
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects, IntrWillReturn]>;

// Issue a prefetch from the GPU to the host. This will be lowered to an
// appropriate call in Kitusne's runtime. The function takes a pointer to
// an opaque stream on which to issue the prefetch, or null to indicate that a
// new stream should be created. Returns the pointer to the stream - if one was
// given - or a new stream that was created. A pointer to the buffer to prefetch
// is also provided. The number of bytes to prefetch may be -1 which indicates
// that the runtime is required to determine the number of bytes to prefetch.
def int_kitrt_prefetch_host
    : Intrinsic<[llvm_ptr_ty], // stream
                [llvm_i8_ty,   // tapir target id
                 llvm_ptr_ty,  // pointer to memory to prefetch
                 llvm_i64_ty,  // number of bytes to prefetch
                 llvm_ptr_ty], // stream
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Set a fixed number of threads per block to use for all kernels launched by
// kitsune's runtime.
def int_kitrt_set_fixed_tpb
    : Intrinsic<[],
                [llvm_i8_ty,   // tapir target id
                 llvm_i32_ty], // threads per block
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOnly,
                 IntrHasSideEffects]>;

// Set the maximum number of threads per block to use for all kernels launched
// by kitsune's runtime.
def int_kitrt_set_max_tpb
    : Intrinsic<[],
                [llvm_i8_ty,   // tapir target id
                 llvm_i32_ty], // threads per block
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOnly,
                 IntrHasSideEffects]>;

// Synchronize a thread stream.
def int_kitrt_sync_stream
    : Intrinsic<[],
                [llvm_i8_ty,   // tapir target id
                 llvm_ptr_ty], // stream
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Get a pointer to a symbol in the device. This is typically used to get the
// address of a global variable, but in principle, it could be used with any
// global value.
def int_kitrt_symbol_device_ptr
    : Intrinsic<[llvm_ptr_ty], // device pointer
                [llvm_i8_ty,   // tapir target id
                 llvm_ptr_ty,  // fat binary
                 llvm_ptr_ty], // name
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly]>;

// Copy a symbol from host to device
def int_kitrt_symbol_memcpy_device
    : Intrinsic<[],
                [llvm_i8_ty,   // tapir target id
                 llvm_ptr_ty,  // pointer to symbol on the device
                 llvm_ptr_ty,  // pointer to symbol on the host
                 llvm_i64_ty], // number of bytes to copy
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Copy a symbol from device to host
def int_kitrt_symbol_memcpy_host
    : Intrinsic<[],
                [llvm_i8_ty,   // tapir target id
                 llvm_ptr_ty,  // pointer to symbol on the host
                 llvm_ptr_ty,  // pointer to symbol on the device
                 llvm_i64_ty], // number of bytes to copy
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;
