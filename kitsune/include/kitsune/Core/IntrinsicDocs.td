//===- IntrinsicDocs.td - Docs for Kitsune's intrinsics ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Documentation for Kitsune-specific intrinsics. These are used to generate
// the HTML documentation.
//
//===----------------------------------------------------------------------===//

// The category to which the intrinsic belongs. This is only used when
// organizing the documentation.
class KitDocCategory {
  // The text to be used in the section header when generating the
  // documentation.
  string Header;

  // The text for the introductory paragraph for the section containing
  // intrinsics in this category.
  code Intro;
}

// Memory allocation intrinsics.
def KitDocCatMemAlloc : KitDocCategory {
  let Header = "Memory Allocation Intrinsics";
  let Intro = [{
These intrinsics specifically allocate and free
:doc:`mobile buffers <MemoryManagement>`.
}];
}

// Intrinsics shared by Kitsune's various runtimes.
def KitDocCatCommon : KitDocCategory {
  let Header = "Common Intrinsics";
  let Intro = [{
These intrinsics correspond to operations that are likely to be supported by
several tapir targets and their associated runtimes.
}];
}

// Intrinsics used by Ktsune's multithreaded CPU runtimes.
def KitDocCatThreading : KitDocCategory {
  let Header = "Threading Intrinsics";
  let Intro = [{
These intrinsics are generally used by tapir targets that lower to a
multithreaded runtime such as OpenMP or Qthreads. These generally operate by
splitting iterations of a :ref:`tapir loop <glossary-tapir-loop>` across threads
on a CPU.
}];
}

// Intrinsics used by the GPU-centric tapir targets.
def KitDocCatGPU : KitDocCategory {
  let Header = "GPU-centric Intrinsics";
  let Intro = [{
These intrinsics are only be used by the GPU-centric tapir targets,
:ref:`cuda <tapir-targets-cuda>` and :ref:`hip <tapir-targets-hip>`.
}];
}

// Documentation for the parameter of an intrinsic.
class KitDocParam<string NameParam, string DescrParam> {
  // The name of the parameter.
  string Name = NameParam;

  // Description of the parameter.
  string Descr = DescrParam;
}

// Documentation for a Kitsune-specific intrinsic.
class KitDocIntrinsic {
  // The category to which this intrinsic belongs. This is only used when
  // generating the HTML documentation.
  KitDocCategory Category;

  // Documentation for the intrinsic function parameters. This must, obviously,
  // be kept consistent with the intrinsics signature.
  list<KitDocParam> ParamDocs;

  // Description of the intrinsic.
  code Descr;
}

def KitMobileAllocDoc : KitDocIntrinsic {
  let Category = KitDocCatMemAlloc;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">,
      KitDocParam<"size", "The number of bytes to allocate">
  ];
  let Descr = [{
Allocate a mobile buffer of `size` bytes. The lowering of this intrinsic may
depend on:

  - The tapir target(s) that have been enabled
  - The uses of the allocated buffer

.. note::
  As of April 2026, uses of the allocated buffer are *not* considered - only
  the primary tapir target is used to decide how the intrinsic is lowered.

The allocated memory must be freed using the :ref:`llvm.kit.mobile.free`
intrinsic.
}];
}

def KitMobileFreeDoc : KitDocIntrinsic {
  let Category = KitDocCatMemAlloc;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">,
      KitDocParam<"buf", "The mobile buffer to be freed">
  ];
  let Descr = [{
Free a mobile buffer that was previously allocated using
:ref:`llvm.kit.mobile.alloc`.
}];
}

def KitMobileInitDoc : KitDocIntrinsic {
  let Category = KitDocCatMemAlloc;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">,
      KitDocParam<"buf", "The mobile buffer to initialize">,
      KitDocParam<"n", "The number of elements in the array">,
      KitDocParam<"v", "The initializer">,
      KitDocParam<"size", "If :code:`v` is a pointer, the size of the pointee">,
  ];
  let Descr = [{
Initialize a mobile buffer. This is most useful when the buffer is a
contiguous array and each element of the array needs to be initialized with a
certain value.

The exactly lowering of this intrinsic is not specified. It may be lowered to a
loop in LLVM IR, or may call an external function. If :code:`tt` is a
GPU-centric tapir target, the initialization may be performed on the GPU, but
this is not guaranteed.

.. note::
  Currently, only the following types are supported for the initializer
  :code:`v`: :code:`i8 i16 i32 i64 float double`. In the future, we may also
  support passing a pointer to the value in cases where the initializer has
  a non-scalar type - such as a struct.
}];
}

def KitInitializeDoc : KitDocIntrinsic {
  let Category = KitDocCatCommon;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">
  ];
  let Descr = [{
Initialize Kitsune's runtime(s). This will initialize the runtime specific to
the tapir target `tt`, as well as the common runtime shared by all tapir
targets.
}];
}

def KitFinalizeDoc : KitDocIntrinsic {
  let Category = KitDocCatCommon;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">
  ];
  let Descr = [{
Finalize Kitsune's runtime(s). This will initialize the runtime specific to
the tapir target `tt`, as well as the common runtime shared by all tapir
targets.
}];
}

def KitEnableVerboseDoc : KitDocIntrinsic {
  let Category = KitDocCatCommon;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">,
      KitDocParam<"enable", "Non-zero to enable verbose mode, 0 to disable it">
  ];
  let Descr = [{
Enable/disable verbose mode in Kitsune's runtime.

.. note::
  This intrinsic does not take a tapir target id because verbose mode is common
  to all tapir targets. In the future, we may support enabling verbose mode on
  select tapir targets. But this is only useful for multi-target compilation
  and execution. This is not yet supported.
}];
}

def KitCPUNumThreadsDoc : KitDocIntrinsic {
  let Category = KitDocCatThreading;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">
  ];
  let Descr = [{
Get the number of CPU threads available on the system. This will return the same
value every time it is called. The actual value that is returned will depend on
the tapir target in use.

In the future, this intrinsic could be modified to take one or more arguments
that would return a number of threads that is most suitable for the loop that
will be executed in parallel.
}];
}

def KitLaunchThreadsDoc : KitDocIntrinsic {
  let Category = KitDocCatThreading;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">,
      KitDocParam<"threadFunc", "The function executed on each thread">,
      KitDocParam<"beg", "Start value of the loop induction variable">,
      KitDocParam<"end", "Trip count of the loop">,
      KitDocParam<"grainSize", "The grain size (currently unused)">,
      KitDocParam<"argBundle", "Argument bundle passed to `threadFunc`">
  ];
  let Descr = [{
Launch a parallel loop on one or more CPU threads.

Currently, `beg` must be 0. `grainSize` is not used and may be removed in the
future. `threadFunc` runs on each thread and handles a subset of the loop
iterations. The actual signature of the function will depend on the tapir target
that calls this, but will generally take at least a start index, an end index
and the `argBundle`. `argBundle` is a pointer to a struct containing values that
will be used by `thrdFunc`. The struct is specific to a given parallel loop and
will typically consist of the entities used in the body of the loop.

.. code :: c++

  parallel_for(size_t i = 0; i < n; ++i)
      dst[i] = src[i]

For instance, in the loop above, `argBundle` will be a pointer to a struct
consisting of two pointers, `dst` and `src`.

This blocks until all threads launched by the intrinsic have finished executing.
For the asynchronous equivalent of this intrinsic, see
:ref:`llvm.kit.async.cpu.threads.launch`.
}];
}

def KitAsyncLaunchThreadsDoc : KitDocIntrinsic {
  let Category = KitDocCatThreading;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">,
      KitDocParam<"threadFunc", "The function executed on each thread">,
      KitDocParam<"beg", "Start value of the loop induction variable">,
      KitDocParam<"end", "Trip count of the loop">,
      KitDocParam<"grainSize", "The grain size (currently unused)">,
      KitDocParam<"argBundle", "Argument bundle passed to `threadFunc`">
  ];
  let Descr = [{
Launch a parallel loop on one or more CPU threads. This is non-blocking and
will return immediately.

Returns a pointer to an opaque thread context object. This object must be passed
to :ref:`llvm.kit.cpu.threads.sync` in order to synchronize the threads. It is
the caller's responsibility to ensure that any memory that is used by
`threadFunc` is not used until after all threads launched here have finished
executing i.e. after the threads have been synchronized.

Currently, `beg` must be 0. `grainSize` is not used and may be removed in the
future. `threadFunc` runs on each thread and handles a subset of the loop
iterations. The actual signature of the function will depend on the tapir target
that calls this, but will generally take at least a start index, an end index
and the `argBundle`. `argBundle` is a pointer to a struct containing values that
will be used by `thrdFunc`. The struct is specific to a given parallel loop and
will typically consist of the entities used in the body of the loop.

.. code :: c++

  parallel_for(size_t i = 0; i < n; ++i)
      dst[i] = src[i]

For instance, in the loop above, `argBundle` will be a pointer to a struct
consisting of two pointers, `dst` and `src`.

For the synchronous, i.e. blocking, equivalent of this intrinsic, see
:ref:`llvm.kit.cpu.threads.launch`.
}];
}

def KitSyncThreadsDoc : KitDocIntrinsic {
  let Category = KitDocCatThreading;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
    KitDocParam<"ctx", "The thread context to synchronize">
  ];
  let Descr = [{
Block until all threads launched by an earlier call to
:ref:`llvm.kit.async.cpu.threads.launch` have finished executing. `ctx` must be
the context object returned by that earlier call.
}];
}

def KitAsyncMemcpyDToHDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
    KitDocParam<"dest", "Pointer to the destination buffer (on the host)">,
    KitDocParam<"source", "Pointer to the source buffer (on the device)">,
    KitDocParam<"n", "The number of bytes to copy">,
    KitDocParam<"stream", "The stream on which to carry out the copy">
  ];
  let Descr = [{
Copy data from a device (usually a GPU) to the host.

This is a non-blocking call i.e. it will return immediately - potentially before
all the data has been copied. For the blocking equivalent of this, see
:ref:`llvm.kit.gpu.memcpy.dtoh`.
}];
}

def KitAsyncMemcpyHToDDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
    KitDocParam<"dest", "Pointer to the destination buffer (on the device)">,
    KitDocParam<"source", "Pointer to the source buffer (on the host)">,
    KitDocParam<"n", "The number of bytes to copy">,
    KitDocParam<"stream", "The stream on which to carry out the copy">
  ];
  let Descr = [{
Copy data from the host to a device (usually a GPU).

This is a non-blocking call i.e. it will return immediately - potentially before
all the data has been copied. For the blocking equivalent of this, see
:ref:`llvm.kit.gpu.memcpy.htod`.
}];
}

def KitMemcpyDToHDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
    KitDocParam<"dest", "Pointer to the destination buffer (on the host)">,
    KitDocParam<"source", "Pointer to the source buffer (on the device)">,
    KitDocParam<"n", "The number of bytes to copy">,
  ];
  let Descr = [{
Copy data from a device (usually a GPU) to the host.

This is a blocking call. For the non-blocking equivalent of this, see
:ref:`llvm.kit.async.gpu.memcpy.dtoh`.
}];
}

def KitMemcpyHToDDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
    KitDocParam<"dest", "Pointer to the destination buffer (on the device)">,
    KitDocParam<"source", "Pointer to the source buffer (on the host)">,
    KitDocParam<"n", "The number of bytes to copy">,
  ];
  let Descr = [{
Copy data from the host to a device (usually a GPU).

This is a blocking call. For the non-blocking equivalent of this, see
:ref:`llvm.kit.async.gpu.memcpy.htod`.
}];
}

def KitAsyncPrefetchDToHDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
    KitDocParam<"ptr", "Pointer to the buffer (on the device) to prefetch">,
    KitDocParam<"n", "The number of bytes to prefetch">,
    KitDocParam<"stream", "The stream on which to carry out the prefetch">
  ];
  let Descr = [{
Prefetch data from a device (usually a GPU) to the host. This will be lowered to
a call to a function in Kitsune's runtime.

Returns the stream on which the prefetch was carried out. If `stream` is not
`nullptr`, this will be returned. Otherwise, a new stream will be created and
returned. `n` may be -1, in which case the number of bytes to prefetch will be
determined by the runtime.

This is a non-blocking intrinsic and may return before all the data has been
prefetched.
}];
}

def KitAsyncPrefetchHToDDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
    KitDocParam<"ptr", "Pointer to the buffer (on the host) to prefetch">,
    KitDocParam<"n", "The number of bytes to prefetch">,
    KitDocParam<"stream", "The stream on which to carry out the prefetch">
  ];
  let Descr = [{
Prefetch data from the host to a device (usually a GPU). This will be lowered to
a call to a function in Kitsune's runtime.

Returns the stream on which the prefetch was carried out. If `stream` is not
`nullptr`, this will be returned. Otherwise, a new stream will be created and
returned. `n` may be -1, in which case the number of bytes to prefetch will be
determined by the runtime.

This is a non-blocking intrinsic and may return before all the data has been
prefetched.
}];
}

def KitGPUEnableKernelLaunchRefinementDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">,
      KitDocParam<"enable", "Non-zero to enable kernel launch refinement, 0 to disable it">
  ];
  let Descr = [{
Enable launch refinement. Setting this will cause Kitsune's runtime to calculate
a kernel's launch parameters based on some heuristic. In the case of NVIDIA
GPU's, this is an occupancy-based heuristic provided by Cuda.

.. note::
  As of May 2025, this is only implemented in kitsune's cuda runtime. If this is
  implemented for kitsune's hip runtime as well, this comment should be updated.
}];
}

def KitGPUEnableXnackDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">,
      KitDocParam<"enable", "Non-zero to enable xnack, 0 to disable it">
  ];
  let Descr = [{
Enable xnack in Kitsune's runtime. This is only relevant for the
:ref:`hip <tapir-targets-hip>` tapir target and its associated Kitsune runtime.
}];
}

def KitGPUEnableYAxisKernelLaunchDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">,
      KitDocParam<"enable", "Non-zero to enable the y-axis launch pattern, 0 to disable it">
  ];
  let Descr = [{
Enable the Y-axis launch pattern for all GPU kernels.

.. note::
  As of May 2025, this is has only been implemented for the
  :ref:`hip <tapir-targets-hip>` tapir target.
}];
}

def KitGPUSymbolAddressDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
    KitDocParam<"deviceCode", "The global variable containing compiled device code">,
    KitDocParam<"name", "The name of the symbol">
  ];
  let Descr = [{
Get the address of a symbol on a device (typically a GPU). This is typically
used to get a pointer to a global variable, but it could, in principle, be used
to get the address of any global value.
}];
}

def KitGPUTIDXDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
  ];
  let Descr = [{
Get the thread index in the `x` dimension in a GPU kernel. In a
multi-dimensional kernel launch, this will be the fastest-changing index.
This is lowered to one of the following intrinsics:

.. csv-table::
  :header: "", ""

  NVPTX, `nvvm.read.ptx.sreg.tid.x`
  AMDGPU, `amdgcn.workitem.id.x`
}];
}

def KitGPUTIDYDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
  ];
  let Descr = [{
Get the thread index in the `y` dimension in a GPU kernel. In a
multi-dimensional kernel launch, this will be the second-fastest-changing index.
This is lowered to one of the following intrinsics:

.. csv-table::
  :header: "", ""

  NVPTX, `nvvm.read.ptx.sreg.tid.y`
  AMDGPU, `amdgcn.workitem.id.y`
}];
}

def KitGPUTIDZDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
  ];
  let Descr = [{
Get the thread index in the `z` dimension in a GPU kernel. In a 3-dimensional
kernel launch, this will be the slowest-changing index. This is lowered to one
of the following intrinsics:

.. csv-table::
  :header: "", ""

  NVPTX, `nvvm.read.ptx.sreg.tid.z`
  AMDGPU, `amdgcn.workitem.id.z`
}];
}

def KitGPUBlockIDXDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
  ];
  let Descr = [{
Get the block index in the `x` dimension in a GPU kernel. In a
multi-dimensional kernel launch, this will be the fastest-changing index.
This is lowered to one of the following intrinsics:

.. csv-table::
   :header: "", ""

   NVPTX, `nvvm.read.ptx.sreg.ctaid.x`
   AMDGPU, `amdgcn.workgroup.id.x`
}];
}

def KitGPUBlockIDYDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
  ];
  let Descr = [{
Get the block index in the `y` dimension in a GPU kernel. In a
multi-dimensional kernel launch, this will be the second-fastest-changing index.
This is lowered to one of the following intrinsics:

.. csv-table::
   :header: "", ""

   NVPTX, `nvvm.read.ptx.sreg.ctaid.y`
   AMDGPU, `amdgcn.workgroup.id.y`
}];
}

def KitGPUBlockIDZDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
  ];
  let Descr = [{
Get the block index in the `z` dimension in a GPU kernel. In a 3-dimensional
kernel launch, this will be the slowest-changing index. This is lowered to one
of the following intrinsics:

.. csv-table::
   :header: "", ""

   NVPTX, `nvvm.read.ptx.sreg.ctaid.z`
   AMDGPU, `amdgcn.workgroup.id.z`
}];
}

def KitGPUBlockSizeXDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
  ];
  let Descr = [{
Get the number of blocks in the `x` dimension in a GPU kernel. In a
multi-dimensional kernel launch, this will be the fastest-changing index.
This is lowered to one of the following:

.. csv-table::
   :header: "", ""

   NVPTX, `nvvm.read.ptx.sreg.ntid.x`
   AMDGPU, `__ockl_get_local_size(0)`

Note that in AMDGPU kernels, the intrinsic is lowered to a function from a
vendor-provided bitcode file, which is called with a specific argument.
}];
}

def KitGPUBlockSizeYDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
  ];
  let Descr = [{
Get the number of blocks in the `y` dimension in a GPU kernel. In a
multi-dimensional kernel launch, this will be the second-fastest-changing index.
This is lowered to one of the following:

.. csv-table::
   :header: "", ""

   NVPTX, `nvvm.read.ptx.sreg.ntid.y`
   AMDGPU, `__ockl_get_local_size(1)`

Note that in AMDGPU kernels, the intrinsic is lowered to a function from a
vendor-provided bitcode file, which is called with a specific argument.
}];
}

def KitGPUBlockSizeZDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
  ];
  let Descr = [{
Get the number of blocks in the `z` dimension in a GPU kernel. In a
3-dimensional kernel launch, this will be the slowest-changing index. This is
lowered to one of the following:

.. csv-table::
   :header: "", ""

   NVPTX, `nvvm.read.ptx.sreg.ntid.z`
   AMDGPU, `__ockl_get_local_size(2)`

Note that in AMDGPU kernels, the intrinsic is lowered to a function from a
vendor-provided bitcode file, which is called with a specific argument.
}];
}

def KitGPUGridSizeXDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
  ];
  let Descr = [{
Get the number of grids in the `x` dimension in a GPU kernel. In a
multi-dimensional kernel launch, this will be the fastest-changing index.
This is lowered to one of the following:

.. csv-table::
   :header: "", ""

   NVPTX, `nvvm.read.ptx.sreg.nctaid.x`
   AMDGPU, `__ockl_get_global_size(0)`

Note that in AMDGPU kernels, the intrinsic is lowered to a function from a
vendor-provided bitcode file, which is called with a specific argument.
}];
}

def KitGPUGridSizeYDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
  ];
  let Descr = [{
Get the number of grids in the `y` dimension in a GPU kernel. In a
multi-dimensional kernel launch, this will be the second-fastest-changing index.
This is lowered to one of the following:

.. csv-table::
   :header: "", ""

   NVPTX, `nvvm.read.ptx.sreg.nctaid.y`
   AMDGPU, `__ockl_get_global_size(1)`

Note that in AMDGPU kernels, the intrinsic is lowered to a function from a
vendor-provided bitcode file, which is called with a specific argument.
}];
}

def KitGPUGridSizeZDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
    KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
  ];
  let Descr = [{
Get the number of grids in the `z` dimension in a GPU kernel. In a
3-dimensional kernel launch, this will be the slowest-changing index. This is
lowered to one of the following:

.. csv-table::
   :header: "", ""

   NVPTX, `nvvm.read.ptx.sreg.nctaid.z`
   AMDGPU, `__ockl_get_global_size(2)`

Note that in AMDGPU kernels, the intrinsic is lowered to a function from a
vendor-provided bitcode file, which is called with a specific argument.
}];
}

def KitAsyncGPULaunchKernelDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">,
      KitDocParam<"devCode", "The global variable containing the compiled kernel">,
      KitDocParam<"name", "The name of the kernel to launch">,
      KitDocParam<"tcZ", "Trip count of the loop at depth 3, or 0 if there is no such loop">,
      KitDocParam<"tcY", "Trip count of the loop at depth 2, or 0 if there is no such loop">,
      KitDocParam<"tcX", "Trip count of the loop at depth 1">,
      KitDocParam<"tpb", "The threads per block to use when launching the kernel">,
      KitDocParam<"props", "Pre-computed properties of the kernel being launched">,
      KitDocParam<"stream", "The stream on which to launch the kernel">,
      KitDocParam<"extra...", "Additional arguments required by the GPU kernel">
  ];
  let Descr = [{
Launch a GPU kernel on a specific stream. The kernel will have been compiled
from a nest of parallel loops with depth in the range [1,3] inclusive.

This will return the stream on which the kernel was launched. This will be the
same as `stream` if `stream` was not `nullptr`. If `stream` is `nullptr`, a
suitable stream will be created and that stream will be returned. The stream is
a pointer to an opaque object and must not be examined directly.

The variable number of arguments are additional arguments required by the
kernel. These will typically be entities used within the loop nest that has been
compiled to a GPU kernel.

.. code :: c++

  parallel_for(size_t i = 0; i < n; ++i)
      dst[i] = src[i]

For instance, in the loop above, `dst` and `src` will be arguments to this
intrinsic since they are used in the body of the parallel loop.

This is a non-blocking intrinsic and will return immediately. The returned
stream can be used to synchronize the GPU i.e. to block until all kernels that
were launched on the stream have finished executing. This is done by passing the
stream to the :ref:`llvm.kit.gpu.stream.sync` intrinsic. It is the callers
responsibility to ensure that the GPU has finished executing before any memory
modified by the kernel is used - either on the host or by another GPU kernel.
}];
}

def KitGPUSyncStreamDoc : KitDocIntrinsic {
   let Category = KitDocCatGPU;
   let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">,
      KitDocParam<"stream", "The stream on to synchronize">,
   ];
   let Descr = [{
Synchronize a GPU stream. This will block until all GPU kernels launched on
the stream have finished executing.
}];
}

def KitGPUGetStreamDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling the intrinsic">
  ];
  let Descr = [{
Get a new GPU stream. Returns a pointer to the opaque stream object. This
pointer must not be used to examine the returned object directly.
}];
}

def KitGPUSetFixedTPBDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
      KitDocParam<"tpb", "The threads per block value">,
  ];
  let Descr = [{
Set a fixed number of threads per block to use when launching kernels generated
by the tapir target `tt`.
}];
}

def KitGPUSetMaxTPBDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
      KitDocParam<"tpb", "The threads per block value">,
  ];
  let Descr = [{
Set the maximum number of threads per block to use when launching kernels
generated by the tapir target `tt`.
}];
}

def KitGPURegisterDevCodeDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
      KitDocParam<"data", "Pointer to the device code to register">,
  ];
  let Descr = [{
Register a global variable containing device code. This will nearly always be
called in a global constructor for Kitsune's runtime. Returns an opaque handle
that can be used to register any global variables used in the device code, and
other things.
}];
}

def KitGPURegisterDevCodeEndDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
      KitDocParam<"handle", "The opaque handle returned by :ref:`llvm.kit.gpu.register.devcode`">,
  ];
  let Descr = [{
Signal that all global variables in the device code have been registered. This
will nearly always be called in a global constructor for Kitsune's runtime. This
is currently only used with the :ref:`cuda <tapir-targets-cuda>` tapir target.
}];
}

def KitGPURegisterGlobalDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
      KitDocParam<"handle", "The opaque handle returned by :ref:`llvm.kit.gpu.register.devcode`">,
      KitDocParam<"hostAddr", "The address of the global variable on the host. The device-side global variable will be initialized with the value on the host">,
      KitDocParam<"hostName", "Name of the global in host code">,
      KitDocParam<"deviceName", "Name of the global in device code">,
      KitDocParam<"size", "Size, in bytes, of the global">,
      KitDocParam<"isExternal", "Is the global externally visible">,
      KitDocParam<"isConstant", "Is the global constant">,
  ];
  let Descr = [{
Register a global variable that is present in device code that was previously
registered with a call to :ref:`llvm.kit.gpu.register.devcode`. This will nearly
always be called from a global constructor for Kitsune's runtime.
}];
}

def KitGPURegisterGlobalManagedDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
      KitDocParam<"handle", "The opaque handle returned by :ref:`llvm.kit.gpu.register.devcode`">,
      KitDocParam<"newAddr", "Address of pointer variable that will eventually contain the address, in UVM, for the global that will be allocated by this function">,
      KitDocParam<"hostAddr", "Address of the global in host code">,
      KitDocParam<"deviceName", "Name of the global in device code">,
      KitDocParam<"size", "Size, in bytes, of the global variable">,
      KitDocParam<"align", "Requested alignment, in bytes, of the memory to be allocated">,
      KitDocParam<"isExternal", "Is the global externally visible">,
      KitDocParam<"isConstant", "Is the global constant">,
  ];
  let Descr = [{
Register a global variable that is present in device code that was previously
previously registered with a call to :ref:`llvm.kit.gpu.register.devcode`. This
will allocate space for the global variable in UVM and return a pointer to that
allocated memory via the `newAddr` out variable. This will nearly always be
called from a global constructor for Kitsune's runtime.
}];
}

def KitGPUUnregisterDevCodeDoc : KitDocIntrinsic {
  let Category = KitDocCatGPU;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
      KitDocParam<"handle", "The opaque handle returned by :ref:`llvm.kit.gpu.register.devcode`">,
  ];
  let Descr = [{
Unregister device code that was previously registered with
:ref:`llvm.kit.gpu.register.devcode`.
}];
}

def KitReduce0Doc : KitDocIntrinsic {
  let Category = KitDocCatCommon;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
      KitDocParam<"result", "Pointer to the result of the reduction">,
      KitDocParam<"size", "The size (in bytes) of the reduced result">,
      KitDocParam<"value", "The value to reduce into `result`">,
      KitDocParam<"unit", "The unit value for the reducer">,
      KitDocParam<"reducer", "The reducer function">,
      KitDocParam<"extra...", "Additional parameters to pass to `reducer`">
  ];
  let Descr = [{
Reduces a 0-dimensional array, i.e. a scalar, using a given reducer function.
This is an overloaded intrinsic that can be specialized for any type of
`value` and `unit`.

The code below shows how this intrinsic might be declared in C++ (here, the
`.`'s that would appear in the name of the intrinsic have been replaced with
`_`).

.. code :: c++

  template <typename T, typename ExtraArgs...>
  void kit_reduce_0(int32_t tt,
                    T *result,
                    int32_t size,
                    T value,
                    T unit,
                    std::function<void(void *result, T value, ExtraArgs...)> reducer,
                    ExtraArgs...)

Here, the types of the variadic `ExtraArgs` accepted by the `reducer` must
exactly match the values provided to this intrinsic. The example below shows how
a `times` reducer might be used with this intrinsic.

.. code :: kitll

  define void @mul(ptr %res, double %v) {
    %1 = load double, ptr %res
    %2 = fadd double %1, %v
    store double %2, ptr %res
    ret void
  }

  define void @f(double %v) {
    %res = alloca double
    call void(i32, ptr, i32, double, double, ptr, ...) @llvm.kit.reduce.0(i32 2, ptr %res, i32 8, double %v, double 1.0, ptr @mul)
    ret void
  }

Here, the unit value is 1.0 since 1 is the identity value for multiplication. A
basic lowering for the intrinsic call above might be as shown below

.. code :: kitll

  call void @mul(ptr %res, double %v)

Only the pointer to the result and the value to be reduced are used here. The
`size` and `unit` arguments are not used. Those arguments are used when the
intrinsic is used in a tapir loop. In this case, the tapir loop and the
intrinsic call are used to transform the loop for parallel execution. See the
documentation for the `kit-prepare` pass for details.
}];
}

def KitReduce1DDoc : KitDocIntrinsic {
  let Category = KitDocCatCommon;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
      KitDocParam<"result", "Pointer to the result of the reduction">,
      KitDocParam<"size", "The size (in bytes) of the reduced result">,
      KitDocParam<"array", "The array containing the values to reduce">,
      KitDocParam<"n", "The number of elements in `array`">,
      KitDocParam<"unit", "The unit value for the reducer">,
      KitDocParam<"reducer", "The reducer function">,
      KitDocParam<"extra...", "Additional parameters to pass to `reducer`">
  ];
  let Descr = [{
Reduces the values in a contiguous, 1-dimensional array using a given reducer.
This is an overloaded intrinsic that an be specialized for any type of array.
The type of each array element must be the same as `unit`.

The code below shows how this intrinsic might be declared in C++ (here, the
`.`'s that would appear in the name of the intrinsic have been replaced with
`_`).

.. code :: c++

  template <typename T, typename ExtraArgs...>
  void kit_reduce_1(int32_t tt,
                    T *result,
                    int32_t size,
                    T *array,
                    int64_t n,
                    T unit,
                    std::function<void(void *result, T value, ExtraArgs...)> reducer,
                    ExtraArgs...)

Here, the types of the variadic `ExtraArgs` accepted by the `reducer` must
exactly match the values provided to this intrinsic. The example below shows how
a `times` reducer might be used with this intrinsic.

.. code :: kitll

  define void @mul(ptr %res, double %v) {
    %1 = load double, ptr %res
    %2 = fadd double %1, %v
    store double %2, ptr %res
    ret void
  }

  define void @f(ptr %array, i64 %n) {
    %res = alloca double
    call void(i32, ptr, i32, double, i64, double, ptr, ...) @llvm.kit.reduce.0(i32 2, ptr %res, i32 8, ptr %array, i64 %n, double 1.0, ptr @mul)
    ret void
  }

The C++ equivalent of the LLVM IR above is shown below.

.. code :: c++

  void mul(double *res, double v) {
    *res *= v;
  }

  void f(double *array, int64_t n) {
    double res;
    kit_reduce_1(&res, sizeof(double), array, n, 1.0, mul);
  }

A straightforward lowering of this intrinsic might be as shown below.

.. code :: c++

  void f(double *array, int64_t n) {
    double res = unit;
    for (int64_t i = 0; i < n; ++i)
      mul(&res, array[i]);
  }

This intrinsic is used by the :ref:`kit-prepare <passes-kit-prepare>` pass to
reduce the array of partial reductions that are computed when a
:ref:`tapir reduction loop <glossary-tapir-reduction-loop>` is transformed for
parallel execution.
}];
}

def KitReduceNumPartialsDoc : KitDocIntrinsic {
  let Category = KitDocCatCommon;
  let ParamDocs = [
      KitDocParam<"tt", "The ID of the tapir target calling this intrinsic">,
      KitDocParam<"n", "The trip count of the tapir reduction loop">
  ];
  let Descr = [{
Get the number of partial reductions to perform when transforming a
:ref:`tapir reduction loop <glossary-tapir-reduction-loop>` for parallel
execution.
}];
}
