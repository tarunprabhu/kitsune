// Check that the global ctor calls the appropriate functions in Kitsune's
// runtime depending on the command line arguments passed.
//
// RUN: %kitxx --tapir=cuda -S -emit-llvm -O2 -o - %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix DEFAULT
//
// Currently, if a max-threads-per-block option is not used, the CudaABI
// nevertheless sets the max to 1024.
//
// DEFAULT: @llvm.global_ctors = appending {{.+}}, ptr @kitcu.ctor{{[^ ]+}},
// DEFAULT: define {{.+}} @kitcu.ctor{{.*}}
// DEFAULT-NOT: call {{.+}}__kitcuda_set_default_threads_per_blk
// DEFAULT: call {{.+}}__kitcuda_set_max_threads_per_blk(i32 1024)
// DEFAULT-NOT: call {{.+}}__kitrt_enable_verbose_mode()
// DEFAULT-DAG: call {{.+}}__kitcuda_initialize()
// DEFAULT-DAG: call {{.+}}__kitcuda_enable_launch_refinement(i8 1)
// DEFAULT-DAG: call {{.+}}__cudaRegisterFatBinary
// DEFAULT: call {{.+}}__cudaRegisterFatBinaryEnd
// DEFAULT: call {{.+}}atexit(ptr nonnull @kitcu.dtor{{[^ ]*}})
// DEFAULT: }
//
// DEFAULT: define {{.+}} @kitcu.dtor{{.*}}
// DEFAULT: call {{.+}} @__cudaUnregisterFatBinary
// DEFAULT: call {{.+}} @__kitcuda_destroy
//
// ----------------------------------------------------------------------------
//
// RUN: %kitxx --tapir=cuda -S -emit-llvm -O2 -o - %s \
// RUN:     --tapir-threads-per-block=77 \
// RUN:     | FileCheck %s -check-prefix TPB
//
// TPB-LABEL: kitcu.ctor{{.*}}
// TPB: call {{.+}}__kitcuda_set_default_threads_per_blk(i32 77)
//
// ----------------------------------------------------------------------------
//
// RUN: %kitxx --tapir=cuda -S -emit-llvm -O2 -o - %s \
// RUN:     --tapir-max-threads-per-block=29 \
// RUN:     | FileCheck %s -check-prefix MTPB
//
// MTPB-LABEL: kitcu.ctor{{.*}}
// MTPB: call {{.+}}__kitcuda_set_max_threads_per_blk(i32 29)
//
// ----------------------------------------------------------------------------
//
// RUN: %kitxx --tapir=cuda -S -emit-llvm -O2 -o - %s \
// RUN:     --tapir-verbose \
// RUN:     | FileCheck %s -check-prefix VERBOSE
//
// RUN: %kitxx --tapir=cuda -S -emit-llvm -O2 -o - %s \
// RUN:     --kitrt-verbose \
// RUN:     | FileCheck %s -check-prefix VERBOSE
//
// VERBOSE-LABEL: kitcu.ctor{{.*}}
// VERBOSE: call {{.+}}__kitrt_enable_verbose_mode()
//
// ----------------------------------------------------------------------------
//
// RUN: %kitxx --tapir=cuda -S -emit-llvm -O2 -o - %s \
// RUN:     -mllvm -cuabi-refine-launches=false \
// RUN:     | FileCheck %s -check-prefix NOREFINE
//
// NOREFINE-LABEL: kitcu.ctor{{.*}}
// NOREFINE: call {{.+}}__kitcuda_enable_launch_refinement(i8 0)
//
// ----------------------------------------------------------------------------

#include <kitsune.h>

void vecadd(double* c, double* a, double* b, size_t n) {
  forall(size_t i = 0; i < n; ++i)
    c[i] = a[i] + b[i];
}
