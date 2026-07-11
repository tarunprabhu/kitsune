// Check that the output of per-thread timers started and stopped multiple times
// is as expected. We use Kitsune's OpenMP runtime because it is guaranteed to
// be built. Also, omp_get_thread_num() returns an integer in
// [0, KIT_NUM_THREADS), so one can reasonably collect times across iterations.
//
// RUN: env KIT_TIMING_FILE=%t.json KIT_NUM_THREADS=3 %exe 2>&1 \
// RUN:     | FileCheck %s --allow-empty --check-prefix=STDERR
// RUN: cat %t.json | FileCheck %s --check-prefix=FILE
//
// -----------------------------------------------------------------------------
// If the file to be written to already exists, its contents will be
// overwritten.
//
// RUN: echo "Contents" > %t.existing.json
// RUN: cat %t.existing.json | FileCheck %s --check-prefix=EXISTING
// RUN: env KIT_TIMING_FILE=%t.existing.json KIT_NUM_THREADS=3 %exe 2>&1 \
// RUN:     | FileCheck %s --allow-empty --check-prefix=STDERR
// RUN: wc -l %t.existing.json | FileCheck %s --check-prefix=FILE_NUM_LINES
// RUN: cat %t.existing.json | FileCheck %s --check-prefix=FILE
//
// -----------------------------------------------------------------------------
// If KIT_TIMING_FILE is set to "-", write timings to stdout.
//
// RUN: env KIT_TIMING_FILE=- KIT_NUM_THREADS=3 %exe 2>&1 \
// RUN:     | FileCheck %s --check-prefix=FILE
//
// -----------------------------------------------------------------------------
// STDERR-NOT: {{^.+$}}
//
// FILE_NUM_LINES: 10
//
// FILE: {
// FILE-NEXT: "kinder": {
// FILE-DAG: "0": [{{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}]
// FILE-DAG: "1": [{{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}]
// FILE-DAG: "2": [{{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}]
// FILE-NEXT: },
// FILE-NEXT: "papagena": {
// FILE-NEXT: "0": [{{[0-9]+}}]
// FILE-NEXT: }
// FILE-NEXT: }
// FILE-NOT: {{^.+$}}
//
// EXISTING: Contents
// EXISTING-NOT: {{^.+$}}
// -----------------------------------------------------------------------------

#include "common/timer.h"
#include "openmp/kitomp.h"

#include <stdlib.h>

unsigned omp_get_thread_num(void);

__attribute__((constructor)) static void ctor(void) { __kitomp_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitomp_finalize(); }

static void thrdFn(int64_t start, int64_t stop, void *args) {
  __kittimer_start(92, omp_get_thread_num(), "kinder");
  __kittimer_stop(92, omp_get_thread_num());
}

int main(int argc, char *argv[]) {
  __kittimer_start(11, 0, "papagena");
  for (unsigned i = 0; i < 3; ++i)
    __kitomp_launch(thrdFn, 0, 3, NULL);
  __kittimer_stop(11, 0);

  return 0;
}
