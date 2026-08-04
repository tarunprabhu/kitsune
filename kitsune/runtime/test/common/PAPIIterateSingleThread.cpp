// REQUIRES: kitsune-papi
//
// Check that the output of PAPI epochs started and stopped multiple times is as
// expected.
//
// RUN: %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=DEFAULT
//
// RUN: env KIT_INSTR_SEPARATE=1 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=SEPARATE
//
// DEFAULT:      {
// DEFAULT-NEXT:   "carmen": {
// DEFAULT-NEXT:     "0": [
// DEFAULT-NEXT:       {"Total cycles": {{[0-9]+}}}
// DEFAULT-NEXT:     ]
// DEFAULT-NEXT:   }
// DEFAULT-NEXT: }
//
// SEPARATE:      {
// SEPARATE-NEXT:   "carmen": {
// SEPARATE-NEXT:     "0": [
// SEPARATE-NEXT:       {"Total cycles": {{[0-9]+}}},
// SEPARATE-NEXT:       {"Total cycles": {{[0-9]+}}},
// SEPARATE-NEXT:       {"Total cycles": {{[0-9]+}}}
// SEPARATE-NEXT:     ]
// SEPARATE-NEXT:   }
// SEPARATE-NEXT: }

#include "common/kitpapi.h"
#include "kitrt.h"

#include "papi.h"

__attribute__((constructor)) static void ctor(void) {
  __kitrt_initialize();
  __kitpapi_initialize(NULL);
}

__attribute__((destructor)) static void dtor(void) {
  __kitpapi_finalize();
  __kitrt_finalize();
}

int main(int argc, char *argv[]) {
  for (unsigned i = 0; i < 3; ++i) {
    KitPAPIEpoch *e = __kitpapi_start("carmen", /*thread=*/0, 1, "cyc");
    __kitpapi_stop(e);
  }

  return 0;
}
