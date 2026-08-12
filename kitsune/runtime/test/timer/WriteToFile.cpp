// If KIT_TIMING_FILE is set to a writable file, check that the timing output is
// written to it.
//
// RUN: env KIT_TIMING_FILE=%t.json KIT_NUM_THREADS=3 KIT_INSTR_SEPARATE=1 \
// RUN:     %exe 2>&1 \
// RUN:     | FileCheck %s --allow-empty --check-prefix=STDERR
// RUN: cat %t.json | FileCheck %s --check-prefix=FILE
//
// -----------------------------------------------------------------------------
// If the file to be written to already exists, its contents will be
// overwritten.
//
// RUN: echo "Contents" > %t.existing.json
// RUN: wc -l %t.existing.json | FileCheck %s --check-prefix=EXISTING_NUM_LINES
// RUN: env KIT_TIMING_FILE=%t.existing.json KIT_NUM_THREADS=3 %exe 2>&1 \
// RUN:     | FileCheck %s --allow-empty --check-prefix=STDERR
// RUN: wc -l %t.existing.json | FileCheck %s --check-prefix=OVR_NUM_LINES
// RUN: cat %t.existing.json | FileCheck %s --check-prefix=OVR
//
// -----------------------------------------------------------------------------
// If KIT_TIMING_FILE is set to "-", write timings to stdout.
//
// RUN: env KIT_TIMING_FILE=- KIT_NUM_THREADS=3 KIT_INSTR_SEPARATE=1 \
// RUN:     %exe \
// RUN:     | FileCheck %s --check-prefix=FILE
// RUN: env KIT_TIMING_FILE=- KIT_NUM_THREADS=3 \
// RUN:     %exe 2>&1 > /dev/null \
// RUN:     | FileCheck %s --allow-empty --check-prefix=STDERR
//
// -----------------------------------------------------------------------------
// STDERR-NOT: {{^.+$}}
//
// FILE_NUM_LINES: 24
//
// FILE:      {
// FILE-NEXT:   "kinder": {
// FILE-NEXT:     "0": [
// FILE-NEXT:       {
// FILE-SAME:         "total": [[K00:[0-9]+]],
// FILE-SAME:         "visits": 1,
// FILE-SAME:         "min": [[K00]],
// FILE-SAME:         "mean": [[K00]],
// FILE-SAME:         "max": [[K00]]
// FILE-SAME:       },
// FILE-NEXT:       {
// FILE-SAME:         "total": [[K01:[0-9]+]],
// FILE-SAME:         "visits": 1,
// FILE-SAME:         "min": [[K01]],
// FILE-SAME:         "mean": [[K01]],
// FILE-SAME:         "max": [[K01]]
// FILE-SAME:       },
// FILE-NEXT:       {
// FILE-SAME:         "total": [[K02:[0-9]+]],
// FILE-SAME:         "visits": 1,
// FILE-SAME:         "min": [[K02]],
// FILE-SAME:         "mean": [[K02]],
// FILE-SAME:         "max": [[K02]]
// FILE-SAME:       }
// FILE-NEXT:     ],
// FILE-NEXT:     "1": [
// FILE-NEXT:       {
// FILE-SAME:         "total": [[K10:[0-9]+]],
// FILE-SAME:         "visits": 1,
// FILE-SAME:         "min": [[K10]],
// FILE-SAME:         "mean": [[K10]],
// FILE-SAME:         "max": [[K10]]
// FILE-SAME:       },
// FILE-NEXT:       {
// FILE-SAME:         "total": [[K11:[0-9]+]],
// FILE-SAME:         "visits": 1,
// FILE-SAME:         "min": [[K11]],
// FILE-SAME:         "mean": [[K11]],
// FILE-SAME:         "max": [[K11]]
// FILE-SAME:       },
// FILE-NEXT:       {
// FILE-SAME:         "total": [[K12:[0-9]+]],
// FILE-SAME:         "visits": 1,
// FILE-SAME:         "min": [[K12]],
// FILE-SAME:         "mean": [[K12]],
// FILE-SAME:         "max": [[K12]]
// FILE-SAME:       }
// FILE-NEXT:     ],
// FILE-NEXT:     "2": [
// FILE-NEXT:       {
// FILE-SAME:         "total": [[K20:[0-9]+]],
// FILE-SAME:         "visits": 1,
// FILE-SAME:         "min": [[K20]],
// FILE-SAME:         "mean": [[K20]],
// FILE-SAME:         "max": [[K20]]
// FILE-SAME:       },
// FILE-NEXT:       {
// FILE-SAME:         "total": [[K21:[0-9]+]],
// FILE-SAME:         "visits": 1,
// FILE-SAME:         "min": [[K21]],
// FILE-SAME:         "mean": [[K21]],
// FILE-SAME:         "max": [[K21]]
// FILE-SAME:       },
// FILE-NEXT:       {
// FILE-SAME:         "total": [[K22:[0-9]+]],
// FILE-SAME:         "visits": 1,
// FILE-SAME:         "min": [[K22]],
// FILE-SAME:         "mean": [[K22]],
// FILE-SAME:         "max": [[K22]]
// FILE-SAME:       }
// FILE-NEXT:     ]
// FILE-NEXT:   },
// FILE-NEXT:   "papagena": {
// FILE-NEXT:     "0": [
// FILE-NEXT:       {
// FILE-SAME:         "total": [[P:[0-9]+]],
// FILE-SAME:         "visits": 1,
// FILE-SAME:         "min": [[P]],
// FILE-SAME:         "mean": [[P]],
// FILE-SAME:         "max": [[P]]
// FILE-SAME:       }
// FILE-NEXT:     ]
// FILE-NEXT:   }
// FILE-NEXT: }
//
// EXISTING_NUM_LINES: 1
//
// OVR_NUM_LINES: 18
// OVR:      {
// OVR-NEXT:   "kinder": {
// OVR-NEXT:     "0": [
// OVR-NEXT:       {
// OVR-SAME:         "total": {{[0-9]+}},
// OVR-SAME:         "visits": 3,
// OVR-SAME:         "min": {{[0-9]+}},
// OVR-SAME:         "mean": {{[0-9]+([.][0-9]+)?}},
// OVR-SAME:         "max": {{[0-9]+}}
// OVR-SAME:       }
// OVR-NEXT:     ],
// OVR-NEXT:     "1": [
// OVR-NEXT:       {
// OVR-SAME:         "total": {{[0-9]+}},
// OVR-SAME:         "visits": 3,
// OVR-SAME:         "min": {{[0-9]+}},
// OVR-SAME:         "mean": {{[0-9]+([.][0-9]+)?}},
// OVR-SAME:         "max": {{[0-9]+}}
// OVR-SAME:       }
// OVR-NEXT:     ],
// OVR-NEXT:     "2": [
// OVR-NEXT:       {
// OVR-SAME:         "total": {{[0-9]+}},
// OVR-SAME:         "visits": 3,
// OVR-SAME:         "min": {{[0-9]+}},
// OVR-SAME:         "mean": {{[0-9]+([.][0-9]+)?}},
// OVR-SAME:         "max": {{[0-9]+}}
// OVR-SAME:       }
// OVR-NEXT:     ]
// OVR-NEXT:   },
// OVR-NEXT:   "papagena": {
// OVR-NEXT:     "0": [
// OVR-NEXT:       {
// OVR-SAME:         "total": [[TOT:[0-9]+]],
// OVR-SAME:         "visits": 1,
// OVR-SAME:         "min": [[TOT]],
// OVR-SAME:         "mean": [[TOT]],
// OVR-SAME:         "max": [[TOT]]
// OVR-SAME:       }
// OVR-NEXT:     ]
// OVR-NEXT:   }
// OVR-NEXT: }
//
// -----------------------------------------------------------------------------

#include "TestHelpers.h"
#include "openmp/kitomp.h"
#include "timer/timer.h"

#include <stdlib.h>

CTOR(RT_TIMER | RT_OPENMP)

static void thrdFn(uint64_t start, uint64_t stop, void *args) {
  KitTimerEpoch *e = __kittimer_start("kinder", __kitomp_thread_id());
  __kittimer_stop(e);
}

int main(int argc, char *argv[]) {
  KitTimerEpoch *e = __kittimer_start("papagena", /*thread=*/0);
  for (unsigned i = 0; i < 3; ++i)
    __kitomp_launch(thrdFn, /*beg=*/0, /*end=*/3, /*args=*/NULL, /*argSize=*/0);
  __kittimer_stop(e);

  return 0;
}
