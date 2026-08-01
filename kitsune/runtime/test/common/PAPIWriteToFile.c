// REQUIRES: kitsune-papi
//
// If KIT_PAPI_FILE is set to a writable file, check that the timing output is
// written to it.
//
// RUN: env KIT_PAPI_FILE=%t.json KIT_NUM_THREADS=3 KIT_INSTR_SEPARATE=1 \
// RUN:     %exe 2>&1 \
// RUN:     | FileCheck %s --allow-empty --check-prefix=STDERR
// RUN: cat %t.json | FileCheck %s --check-prefix=FILE
//
// -----------------------------------------------------------------------------
// If the file to be written to already exists, its contents will be
// overwritten.
//
// RUN: echo "Contents" > %t.existing.json
// RUN: cat %t.existing.json | FileCheck %s --check-prefix=EXISTING
// RUN: env KIT_PAPI_FILE=%t.existing.json KIT_NUM_THREADS=3 %exe 2>&1 \
// RUN:     | FileCheck %s --allow-empty --check-prefix=STDERR
// RUN: wc -l %t.existing.json | FileCheck %s --check-prefix=OVR_NUM_LINES
// RUN: cat %t.existing.json | FileCheck %s --check-prefix=OVR
//
// -----------------------------------------------------------------------------
// If KIT_PAPI_FILE is set to "-", write timings to stdout.
//
// RUN: env KIT_PAPI_FILE=- KIT_NUM_THREADS=3 KIT_INSTR_SEPARATE=1 \
// RUN:     %exe \
// RUN:     | FileCheck %s --check-prefix=FILE
// RUN: env KIT_PAPI_FILE=- KIT_NUM_THREADS=3 \
// RUN:     %exe 2>&1 > /dev/null \
// RUN:     | FileCheck %s --allow-empty --check-prefix=STDERR
//
// -----------------------------------------------------------------------------
// STDERR-NOT: {{^.+$}}
//
// FILE_NUM_LINES: 19
//
// FILE:      {
// FILE-NEXT:   "mercedes": {
// FILE-NEXT:     "0": [
// FILE-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}},
// FILE-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}},
// FILE-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// FILE-NEXT:     ],
// FILE-NEXT:     "1": [
// FILE-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}},
// FILE-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}},
// FILE-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// FILE-NEXT:     ],
// FILE-NEXT:     "2": [
// FILE-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}},
// FILE-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}},
// FILE-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// FILE-NEXT:     ]
// FILE-NEXT:   }
// FILE-NEXT: }
//
// EXISTING: Contents
// EXISTING-NOT: {{^.+$}}
//
// OVR_NUM_LINES: 13
//
// OVR:      {
// OVR-NEXT:   "mercedes": {
// OVR-NEXT:     "0": [
// OVR-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// OVR-NEXT:     ],
// OVR-NEXT:     "1": [
// OVR-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// OVR-NEXT:     ],
// OVR-NEXT:     "2": [
// OVR-NEXT:       {"Instr completed": {{[0-9]+}}, "Total cycles": {{[0-9]+}}}
// OVR-NEXT:     ]
// OVR-NEXT:   }
// OVR-NEXT: }
//
// -----------------------------------------------------------------------------

#include "common/kitpapi.h"
#include "openmp/kitomp.h"

#include "papi.h"

unsigned omp_get_thread_num(void);

__attribute__((constructor)) static void ctor(void) { __kitomp_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitomp_finalize(); }

static void thrdFn(uint64_t start, uint64_t stop, void *args) {
  KitPAPIEpoch *e =
      __kitpapi_start("mercedes", omp_get_thread_num(), 2, "ins", "cyc");
  __kitpapi_stop(e);
}

int main(int argc, char *argv[]) {
  for (unsigned i = 0; i < 3; ++i)
    __kitomp_launch(thrdFn, /*beg=*/0, /*end=*/3, /*args=*/NULL, /*argSize=*/0);
  return 0;
}
