// REQUIRES: kitsune-papi
//
// By default, PAPI counters are written to stderr, not stdout.
//
// RUN: %exe 2>&1 > /dev/null | FileCheck %s --check-prefix=STDERR
// RUN: %exe 2> /dev/null | FileCheck %s --check-prefix=STDOUT --allow-empty
//
// STDERR:     {
// STDERR-NEXT:  "micaela": {
// STDERR-NEXT:    "0": [
// STDERR-NEXT:      {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// STDERR-NEXT:    ]
// STDERR-NEXT:  }
// STDERR-NEXT:}
//
// STDOUT-NOT: {{^.+$}}

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
  KitPAPIEpoch *e = __kitpapi_new("micaela", /*thread=*/0, 2, "ins", "cyc");
  __kitpapi_start(e);
  __kitpapi_stop(e);

  return 0;
}
