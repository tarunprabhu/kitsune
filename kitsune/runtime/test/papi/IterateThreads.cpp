// REQUIRES: kitsune-papi
//
// Check that the output of per-thread PAPI counters that are recorded multiple
// times is as expected. We use Kitsune's OpenMP runtime because it is
// guaranteed to be built. Also, omp_get_thread_num() returns an integer in [0,
// KIT_NUM_THREADS), so one can have reasonable thread IDs.
//
// RUN: env KIT_NUM_THREADS=3 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=DEFAULT
//
// RUN: env KIT_NUM_THREADS=3 KIT_INSTR_SEPARATE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=SEPARATE
//
// DEFAULT:     {
// DEFAULT-NEXT:  "remendado": {
// DEFAULT-NEXT:    "0": [
// DEFAULT-NEXT:      {"Total cycles": {{[0-9]+}}}
// DEFAULT-NEXT:    ],
// DEFAULT-NEXT:    "1": [
// DEFAULT-NEXT:      {"Total cycles": {{[0-9]+}}}
// DEFAULT-NEXT:    ],
// DEFAULT-NEXT:    "2": [
// DEFAULT-NEXT:      {"Total cycles": {{[0-9]+}}}
// DEFAULT-NEXT:    ]
// DEFAULT-NEXT:  }
// DEFAULT-NEXT:}
//
// SEPARATE:     {
// SEPARATE-NEXT:  "remendado": {
// SEPARATE-NEXT:    "0": [
// SEPARATE-NEXT:      {"Total cycles": {{[0-9]+}}},
// SEPARATE-NEXT:      {"Total cycles": {{[0-9]+}}},
// SEPARATE-NEXT:      {"Total cycles": {{[0-9]+}}}
// SEPARATE-NEXT:    ],
// SEPARATE-NEXT:    "1": [
// SEPARATE-NEXT:      {"Total cycles": {{[0-9]+}}},
// SEPARATE-NEXT:      {"Total cycles": {{[0-9]+}}},
// SEPARATE-NEXT:      {"Total cycles": {{[0-9]+}}}
// SEPARATE-NEXT:    ],
// SEPARATE-NEXT:    "2": [
// SEPARATE-NEXT:      {"Total cycles": {{[0-9]+}}},
// SEPARATE-NEXT:      {"Total cycles": {{[0-9]+}}},
// SEPARATE-NEXT:      {"Total cycles": {{[0-9]+}}}
// SEPARATE-NEXT:    ]
// SEPARATE-NEXT:  }
// SEPARATE-NEXT:}

#include "TestHelpers.h"
#include "openmp/kitomp.h"
#include "papi/kitpapi.h"

CTOR(RT_PAPI | RT_OPENMP)

static void thrdFn(uint64_t start, uint64_t stop, void *args) {
  kitrt::KitPAPIEpoch *e =
      __kitpapi_start("remendado", __kitomp_thread_id(), 1, "cyc");
  __kitpapi_stop(e);
}

int main(int argc, char *argv[]) {
  for (unsigned i = 0; i < 3; ++i)
    __kitomp_launch(thrdFn, /*beg=*/0, /*end=*/3, /*args=*/NULL, /*argSize=*/0);
  return 0;
}
