//===- LibFuncs.td - Functions from Kitsune's runtime ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines functions from Kitsune's runtime. These designed to be
// call from the compiler.
//
//===----------------------------------------------------------------------===//

// Base class for the types that may appear in the signatures of Kitsune's
// runtime functions.
class CType<string NameParam> {
  string Name = NameParam;
}

def Void : CType<"void">;
def Bool : CType<"bool">;
def Ptr : CType<"void*">;
def I8 : CType<"int8_t">;
def I16 : CType<"int16_t">;
def I32 : CType<"int32_t">;
def I64 : CType<"int64_t">;
def U8 : CType<"uint8_t">;
def U16 : CType<"uint16_t">;
def U32 : CType<"uint32_t">;
def U64 : CType<"uint64_t">;
def Float : CType<"float">;
def Double : CType<"double">;
def VarArgs : CType<"...">;

// A function from Kitsune's runtime.
class KitFunc<string NameParam, CType RetType, list<CType> ParamTypes> {
  string Name = NameParam;
  CType Ret = RetType;
  list<CType> Params = ParamTypes;
}

// -----------------------------------------------------------------------------
// Runtime library functions common to several, or all, tapir targets.

// void *__kit_instr_start(const char *name, uint64_t thrd)
def kit_instr_start :
    KitFunc<"__kit_instr_start", Ptr, [Ptr, U64]>;

// void __kit_instr_stop(void *epoch)
def kit_instr_stop :
    KitFunc<"__kit_instr_stop", Void, [Ptr]>;

// void *__kitpapi_start(const char *name, uint64_t thrd, uint32_t n, ...)
def kitpapi_start :
    KitFunc<"__kitpapi_start", Ptr, [Ptr, U64, U32, VarArgs]>;

// void __kitpapi_stop(void* epoch)
def kitpapi_stop :
    KitFunc<"__kitpapi_stop", Void, [Ptr]>;

// void *__kitrt_malloc(uint64_t bytes)
def kitrt_malloc :
    KitFunc<"__kitrt_malloc", Ptr, [U64]>;

// void __kitrt_free(void *ptr)
def kitrt_free :
    KitFunc<"__kitrt_free", Void, [Ptr]>;

// void __kitrt_initialize(const KitrtInitOptions *initOpts)
def kitrt_initialize :
    KitFunc<"__kitrt_initialize", Void, [Ptr]>;

// void __kitrt_finalize(const KitrtInitOptions *initOpts)
def kitrt_finalize :
    KitFunc<"__kitrt_finalize", Void, [Ptr]>;

// void *__kitrt_default_mem_alloc(uint64_t bytes)
def kitrt_mem_alloc :
    KitFunc<"__kitrt_default_mem_alloc", Ptr, [U64]>;

// void __kitrt_default_mem_free(void *ptr)
def kitrt_mem_free:
    KitFunc<"__kitrt_default_mem_free", Void, [Ptr]>;

// void __kitrt_mobile_init_bool(void *buf, uint64_t n, bool init)
def kitrt_mobile_init_bool :
    KitFunc<"__kitrt_mobile_init_bool", Void, [Ptr, U64, Bool]>;

// void __kitrt_mobile_init_i8(void *buf, uint64_t n, int8_t init)
def kitrt_mobile_init_i8 :
    KitFunc<"__kitrt_mobile_init_i8", Void, [Ptr, U64, I8]>;

// void __kitrt_mobile_init_i16(void *buf, uint64_t n, int16_t init)
def kitrt_mobile_init_i16 :
    KitFunc<"__kitrt_mobile_init_i16", Void, [Ptr, U64, I16]>;

// void __kitrt_mobile_init_i32(void *buf, uint64_t n, int32_t init)
def kitrt_mobile_init_i32 :
    KitFunc<"__kitrt_mobile_init_i32", Void, [Ptr, U64, I32]>;

// void __kitrt_mobile_init_i64(void *buf, uint64_t n, int64_t init)
def kitrt_mobile_init_i64 :
    KitFunc<"__kitrt_mobile_init_i64", Void, [Ptr, U64, I64]>;

// void __kitrt_mobile_init_float(void *buf, uint64_t n, float init)
def kitrt_mobile_init_float :
    KitFunc<"__kitrt_mobile_init_float", Void, [Ptr, U64, Float]>;

// void __kitrt_mobile_init_double(void *buf, uint64_t n, double init)
def kitrt_mobile_init_double :
    KitFunc<"__kitrt_mobile_init_double", Void, [Ptr, U64, Double]>;

// void __kitrt_mobile_init_from(void *buf, uint64_t n, void *obj,
//                               uint32_t objSize)
def kitrt_mobile_init_from :
    KitFunc<"__kitrt_mobile_init_from", Void, [Ptr, U64, Ptr, U32]>;

// void* __kittimer_start(const char *name, uint64_t thrd)
def kittimer_start :
    KitFunc<"__kittimer_start", Ptr, [Ptr, U64]>;

// uint64_t __kittimer_stop(void *epoch)
def kittimer_stop :
    KitFunc<"__kittimer_stop", U64, [Ptr]>;

// -----------------------------------------------------------------------------
// Runtime library functions for the cuda tapir taret.

// void *__kitcuda_malloc(uint64_t bytes)
def kitcuda_malloc :
    KitFunc<"__kitcuda_malloc", Ptr, [U64]>;

// void __kitcuda_free(void *ptr)
def kitcuda_free :
    KitFunc<"__kitcuda_free", Void, [Ptr]>;

// void*
// __kitcuda_launch_kernel(void* fatbin, const char* name, int64_t tc_x,
//                         int64_t tc_y, int64_t tc_z, int32_t thrds_per_block,
//                         InstrMixInfo* instr_mix, void* stream, void* args[])
def kitcuda_kernel_launch :
    KitFunc<"__kitcuda_launch_kernel", Ptr,
            [Ptr, Ptr, I64, I64, I64, I32, Ptr, Ptr, Ptr]>;

// void __kitcuda_mem_free(void *ptr)
def kitcuda_managed_free :
    KitFunc<"__kitcuda_mem_free", Void, [Ptr]>;

// [[malloc]] void *__kitcuda_mem_alloc_managed(uint64_t bytes)
def kitcuda_managed_malloc :
    KitFunc<"__kitcuda_mem_alloc_managed", Ptr, [U64]>;

// void __kitcuda_memcpy_dtoh(void *hostPtr, void *devicePtr, uint64_t n)
def kitcuda_memcpy_dtoh :
    KitFunc<"__kitcuda_memcpy_dtoh", Void, [Ptr, U64, U64]>;

// void __kitcuda_memcpy_htod(void *devicePtr, void *hostPtr, uint64_t n)
def kitcuda_memcpy_htod :
    KitFunc<"__kitcuda_memcpy_htod", Void, [U64, Ptr, U64]>;

// uint64_t __kitcuda_num_sms()
def kitcuda_num_sms :
    KitFunc<"__kitcuda_num_sms", U64, []>;

// void* __kitcuda_mem_host_prefetch(void *ptr, uint64_t bytes, void *stream)
def kitcuda_prefetch_dtoh :
    KitFunc<"__kitcuda_mem_host_prefetch", Ptr, [Ptr, U64, Ptr]>;

// void* __kitcuda_mem_gpu_prefetch(void *ptr, uint64_t bytes, void *stream)
def kitcuda_prefetch_htod :
    KitFunc<"__kitcuda_mem_gpu_prefetch", Ptr, [Ptr, U64, Ptr]>;

// void* __kitcuda_register_devcode(void* deviceCode)
def kitcuda_register_devcode :
    KitFunc<"__kitcuda_register_devcode", Ptr, [Ptr]>;

// void __kitcuda_register_devcode_end(void* handle)
def kitcuda_register_devcode_end :
    KitFunc<"__kitcuda_register_devcode_end", Void, [Ptr]>;

// void __kitcuda_register_global(
//     void**      handle,    // Pointer returned by __kitcuda_register_devcode
//     void*       hostVar,   // Address of the global on the host
//     char*       hostName,  // Name of the global in host code
//     const char* devName,   // Name of the global in device code
//     uint64_t    size,      // Size, in bytes, of the global
//     int         isExt,     // Is the global externally visible
//     int         isConst)   // Is the global constant
def kitcuda_register_global :
    KitFunc<"__kitcuda_register_global", Void,
            [Ptr, Ptr, Ptr, Ptr, U64, I32, I32]>;

// void __kitcuda_register_managed_global(
//     void**      handle,    // Pointer returned by __kitcuda_register_devcode
//     void**      newAddr,   // Pointer to a chunk of managed memory with size
//                            // \p size and alignment \p align. The runtime
//                            // allocates this managed memory and assigns it
//                            // to \p pointer
//     void*       hostAddr,  // Address of the global on the host
//     const char* name,      // Name of the global in the code object
//     size_t      size,      // Size, in bytes, of the global
//     int         align,     // The variable's alignment
//     int         isExt,     // Is the global externally visible
//     int         isConst)   // Is the global constant
def kitcuda_register_global_managed :
    KitFunc<"__kitcuda_register_global_managed", Void,
            [Ptr, Ptr, Ptr, Ptr, U64, U32, I32, I32]>;

// void __kitcuda_unregister_devcode(void** handle)
def kitcuda_unregister_devcode :
    KitFunc<"__kitcuda_unregister_devcode", Void, [Ptr]>;

// void *__kitcuda_get_thread_stream()
def kitcuda_stream_new :
    KitFunc<"__kitcuda_get_thread_stream", Ptr, []>;

// void __kitcuda_sync_thread_stream(void *stream)
def kitcuda_stream_sync :
    KitFunc<"__kitcuda_sync_thread_stream", Void, [Ptr]>;

// void* __kitcuda_get_global_symbol(void* devcode, const char* name)
def kitcuda_symbol_address :
    KitFunc<"__kitcuda_get_global_symbol", Ptr, [Ptr, Ptr]>;

// -----------------------------------------------------------------------------
// Runtime library functions for the hip tapir taret.

def kithip_enable_xnack :
    KitFunc<"__kithip_enable_xnack", Void, []>;

def kithip_enable_y_axis_launches :
    KitFunc<"__kithip_enable_ylaunch", Void, []>;

// void *__kithip_malloc(uint64_t bytes)
def kithip_malloc :
    KitFunc<"__kithip_malloc", Ptr, [U64]>;

// void __kithip_free(void *ptr)
def kithip_free :
    KitFunc<"__kithip_free", Void, [Ptr]>;

// void*
// __kithip_launch_kernel(void* fatbin, const char* name, int64_t tc_x,
//                        int64_t tc_y, int64_t tc_z, int32_t thrds_per_block,
//                        InstrMixInfo* instr_mix, void* stream, void* args[])
def kithip_kernel_launch :
    KitFunc<"__kithip_launch_kernel", Ptr,
            [Ptr, Ptr, I64, I64, I64, I32, Ptr, Ptr, Ptr]>;

// void __kithip_mem_free(void *ptr)
def kithip_managed_free :
    KitFunc<"__kithip_mem_free", Void, [Ptr]>;

// [[malloc]] void *__kithip_mem_alloc_managed(uint64_t bytes)
def kithip_managed_malloc :
    KitFunc<"__kithip_mem_alloc_managed", Ptr, [U64]>;

// void __kithip_memcpy_dtoh(void *hostPtr, void *devicePtr, uint64_t n)
def kithip_memcpy_dtoh :
    KitFunc<"__kithip_memcpy_dtoh", Void, [Ptr, Ptr, U64]>;

// void __kithip_memcpy_htod(void *hostPtr, void *devicePtr, uint64_t n)
def kithip_memcpy_htod :
    KitFunc<"__kithip_memcpy_htod", Void, [Ptr, Ptr, U64]>;

// uint64_t __kithip_num_cus()
def kithip_num_cus :
    KitFunc<"__kithip_num_cus", U64, []>;

// void* __kithip_mem_host_prefetch(void *ptr, uint64_t bytes, void *stream)
def kithip_prefetch_dtoh :
    KitFunc<"__kithip_mem_host_prefetch", Ptr, [Ptr, U64, Ptr]>;

// void* __kithip_mem_gpu_prefetch(void *ptr, uint64_t bytes, void *stream)
def kithip_prefetch_htod :
    KitFunc<"__kithip_mem_gpu_prefetch", Ptr, [Ptr, U64, Ptr]>;

// void* __kithip_register_devcode(void* deviceCode)
def kithip_register_devcode :
    KitFunc<"__kithip_register_devcode", Ptr, [Ptr]>;

// void __kithip_register_global(
//     void**      handle,    // Pointer returned by __kithip_register_devcode
//     void*       hostVar,   // Address of the global on the host
//     char*       hostName,  // Name of the global in host code
//     const char* devName,   // Name of the global in device code
//     uint64_t    size,      // Size, in bytes, of the global
//     int         isExt,     // Is the global externally visible
//     int         isConst)   // Is the global constant
def kithip_register_global :
    KitFunc<"__kithip_register_global", Void,
            [Ptr, Ptr, Ptr, Ptr, U64, I32, I32]>;

// void __kithip_register_managed_global(
//     void**      handle,    // Pointer returned by __kithip_register_devcode
//     void**      newAddr,   // Pointer to a chunk of managed memory with size
//                            // \p size and alignment \p align. The runtime
//                            // allocates this managed memory and assigns it
//                            // to \p pointer
//     void*       hostAddr,  // Address of the global on the host
//     const char* name,      // Name of the global in the code object
//     size_t      size,      // Size, in bytes, of the global
//     int         align,     // The variable's alignment
//     int         isExt,     // Is the global externally visible
//     int         isConst)   // Is the global constant
def kithip_register_global_managed :
    KitFunc<"__kithip_register_global_managed", Void,
            [Ptr, Ptr, Ptr, Ptr, U64, U32, I32, I32]>;

// void __kithip_unregister_devcode(void** handle)
def kithip_unregister_devcode :
    KitFunc<"__kithip_unregister_devcode", Void, [Ptr]>;

// void *__kithip_get_thread_stream()
def kithip_stream_new :
    KitFunc<"__kithip_get_thread_stream", Ptr, []>;

// void __kithip_sync_thread_stream(void *stream)
def kithip_stream_sync :
    KitFunc<"__kithip_sync_thread_stream", Void, [Ptr]>;

// void* __kithip_get_global_symbol(void* devcode, const char* name)
def kithip_symbol_address :
    KitFunc<"__kithip_get_global_symbol", Ptr, [Ptr, Ptr]>;

// -----------------------------------------------------------------------------
// Runtime library functions for the opencilk tapir taret.

def kitocilk_num_workers :
    KitFunc<"__kitocilk_num_workers", U64, []>;

def kitocilk_worker_id :
    KitFunc<"__kitocilk_worker_id", U64, []>;

// -----------------------------------------------------------------------------
// Runtime library functions for the openmp tapir taret.

def kitomp_num_threads :
    KitFunc<"__kitomp_num_threads", U64, []>;

// void __kitomp_launch(void(*thread_f)(uint64_t, uint64_t, void*),
//                      uint64_t beg, uint64_t end, void* args,
//                      uint32_t argSize);
def kitomp_launch :
    KitFunc<"__kitomp_launch", Void, [Ptr, U64, U64, Ptr, U32]>;

def kitomp_thread_id :
    KitFunc<"__kitomp_thread_id", U64, []>;

// -----------------------------------------------------------------------------
// Runtime library functions for the pthreads tapir taret.

// void* __kitpthr_async_launch(void(*thread_f)(uint64_t, uint64_t, void*),
//                              uint64_t beg, uint64_t end, void* args,
//                              uint32_t argSize);
def kitpthr_async_launch :
    KitFunc<"__kitpthr_async_launch", Ptr, [Ptr, U64, U64, Ptr, U32]>;

def kitpthr_num_threads :
    KitFunc<"__kitpthr_num_threads", U64, []>;

// void __kitpthr_sync(void*)
def kitpthr_sync :
    KitFunc<"__kitpthr_sync", Void, [Ptr]>;

def kitpthr_thread_id :
    KitFunc<"__kitpthr_thread_id", U64, []>;

// -----------------------------------------------------------------------------
// Runtime library functions for the qthreads tapir taret.

def kitqthr_num_workers :
    KitFunc<"__kitqthr_num_workers", U64, []>;

// void __kitqthr_launch(void(*thread_f)(uint64_t, uint64_t, void*),
//                       uint64_t beg, uint64_t end, void* args,
//                       uint32_t argSize);
def kitqthr_launch :
    KitFunc<"__kitqthr_launch", Void, [Ptr, U64, U64, Ptr, U32]>;

def kitqthr_worker_id :
    KitFunc<"__kitqthr_worker_id", U64, []>;
