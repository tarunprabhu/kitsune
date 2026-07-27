// REQUIRES: kitsune-papi
//
// Check that the output of per-thread PAPI counters that are recorded multiple
// times is as expected. We use Kitsune's OpenMP runtime because it is
// guaranteed to be built. Also, omp_get_thread_num() returns an integer in [0,
// KIT_NUM_THREADS), so one can have reasonable thread IDs.
//
// RUN: env KIT_NUM_THREADS=3 %exe 2>&1 | FileCheck %s
//
// CHECK:     {
// CHECK-NEXT:  "remendado": {
// CHECK-NEXT:    "0": [
// CHECK-NEXT:      {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}},
// CHECK-NEXT:      {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}},
// CHECK-NEXT:      {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// CHECK-NEXT:    ],
// CHECK-NEXT:    "1": [
// CHECK-NEXT:      {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}},
// CHECK-NEXT:      {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}},
// CHECK-NEXT:      {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// CHECK-NEXT:    ],
// CHECK-NEXT:    "2": [
// CHECK-NEXT:      {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}},
// CHECK-NEXT:      {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}},
// CHECK-NEXT:      {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:}

#include "common/kitpapi.h"
#include "openmp/kitomp.h"

#include "papi.h"

unsigned omp_get_thread_num(void);

__attribute__((constructor)) static void ctor(void) { __kitomp_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitomp_finalize(); }

static void thrdFn(uint64_t start, uint64_t stop, void *args) {
  KitPAPIEpoch *e = __kitpapi_new("remendado", omp_get_thread_num(),
                                  PAPI_TOT_INS, PAPI_TOT_CYC, 0);
  __kitpapi_start(e);
  __kitpapi_stop(e);
}

int main(int argc, char *argv[]) {
  for (unsigned i = 0; i < 3; ++i)
    __kitomp_launch(thrdFn, /*beg=*/0, /*end=*/3, /*args=*/NULL, /*argSize=*/0);
  return 0;
}
