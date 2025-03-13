// Check that the global ctor calls the appropriate functions in Kitsune's
// runtime depending on the command line arguments passed.
//
// RUN: %kitxx --tapir=hip -S -emit-llvm -O2 -o - %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix DEFAULT
//
// Currently, if a max-threads-per-block option is not used, the HipABI
// nevertheless sets the max to 1024.
//
// DEFAULT: @llvm.global_ctors = appending {{.+}}, ptr @kithip.ctor{{[^ ]+}},
// DEFAULT: define {{.+}} @kithip.ctor{{.*}}
// DEFAULT: call {{.+}}__kithip_enable_xnack()
// DEFAULT-NOT: call {{.+}}__kithip_enable_ylaunch()
// DEFAULT-NOT: call {{.+}}__kithip_set_threads_per_blk
// DEFAULT: call {{.+}}__kithip_set_max_threads_per_blk(i32 1024)
// DEFAULT-NOT: call {{.+}}__kitrt_enable_verbose_mode()
// DEFAULT-DAG: call {{.+}}__kithip_initialize()
// DEFAULT-DAG: call {{.+}}__hipRegisterFatBinary
// DEFAULT: call {{.+}}atexit(ptr nonnull @kithip.dtor{{[^ ]*}})
// DEFAULT: }
//
// FIXME: There is a bug where calling __kithip_destroy raises a segmentation
// fault or some other error which looks like memory corruption bug. As a
// temporary workaround, __kithip_destroy is not called, but it eventually
// should be once the issue is fixed.
//
// DEFAULT: define {{.+}} @kithip.dtor{{.*}}
// DEFAULT: call {{.+}} @__hipUnregisterFatBinary
// DEFAULT-NOT: call {{.+}} @__kithip_destroy
//
// ----------------------------------------------------------------------------
//
// RUN: %kitxx --tapir=hip -S -emit-llvm -O2 -o - %s \
// RUN:     --tapir-threads-per-block=77 \
// RUN:     | FileCheck %s -check-prefix TPB
//
// TPB-LABEL: kithip.ctor{{.*}}
// TPB: call {{.+}}__kithip_set_threads_per_blk(i32 77)
//
// ----------------------------------------------------------------------------
//
// RUN: %kitxx --tapir=hip -S -emit-llvm -O2 -o - %s \
// RUN:     --tapir-max-threads-per-block=29 \
// RUN:     | FileCheck %s -check-prefix MTPB
//
// MTPB-LABEL: kithip.ctor{{.*}}
// MTPB: call {{.+}}__kithip_set_max_threads_per_blk(i32 29)
//
// ----------------------------------------------------------------------------
//
// RUN: %kitxx --tapir=hip -S -emit-llvm -O2 -o - %s \
// RUN:     --tapir-verbose \
// RUN:     | FileCheck %s -check-prefix VERBOSE
//
// RUN: %kitxx --tapir=hip -S -emit-llvm -O2 -o - %s \
// RUN:     --kitrt-verbose \
// RUN:     | FileCheck %s -check-prefix VERBOSE
//
// VERBOSE-LABEL: kithip.ctor{{.*}}
// VERBOSE: call {{.+}}__kitrt_enable_verbose_mode()
//
// ----------------------------------------------------------------------------
//
// RUN: %kitxx --tapir=hip -S -emit-llvm -O2 -o - %s \
// RUN:     -mllvm -hipabi-xnack=false \
// RUN:     | FileCheck %s -check-prefix NOXNACK
//
// NOXNACK-LABEL: kithip.ctor{{.*}}
// NOXNACK-NOT: call {{.+}}__kithip_enable_xnack()
//
// ----------------------------------------------------------------------------
//
// RUN: %kitxx --tapir=hip -S -emit-llvm -O2 -o - %s \
// RUN:     -mllvm -hipabi-y-launch \
// RUN:     | FileCheck %s -check-prefix YLAUNCH
//
// YLAUNCH-LABEL: kithip.ctor{{.*}}
// YLAUNCH: call {{.+}}__kithip_enable_ylaunch()
//
// ----------------------------------------------------------------------------

#include <kitsune.h>

void vecadd(double* c, double* a, double* b, size_t n) {
  forall(size_t i = 0; i < n; ++i)
    c[i] = a[i] + b[i];
}
