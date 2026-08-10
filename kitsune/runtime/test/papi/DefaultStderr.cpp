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

#include "TestHelpers.h"
#include "papi/kitpapi.h"

CTOR(RT_PAPI)

int main(int argc, char *argv[]) {
  KitPAPIEpoch *e = __kitpapi_start("micaela", /*thread=*/0, 2, "ins", "cyc");
  __kitpapi_stop(e);

  return 0;
}
