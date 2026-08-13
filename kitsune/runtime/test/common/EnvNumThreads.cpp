// Check that a warning is emitted if KIT_NUM_THREADS is set to an invalid
// value. The warning should also be emitted if the alternate is set to an
// invalid value.
//
// RUN: env KIT_NUM_THREADS=0 %exe 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env KIT_NUM_THREADS=one %exe 2>&1 | FileCheck %s --check-prefix=DEFAULT
//
// DEFAULT: Ignoring environment variable 'KIT_NUM_THREADS' with invalid value
//
// RUN: env ALTERNATE=0 %exe 2>&1 | FileCheck %s --check-prefix=ALTERNATE
// RUN: env ALTERNATE=one %exe 2>&1 | FileCheck %s --check-prefix=ALTERNATE
//
// ALTERNATE: Ignoring environment variable 'ALTERNATE' with invalid value

#include "TestHelpers.h"
#include "common/utils.h"

#include <cstdlib>

CTOR(RT_COMMON)

int main(int argc, char *argv[]) {
  if (getenv("ALTERNATE"))
    kitrt::getNumThreadsOrCPUs("ALTERNATE");
  else
    kitrt::getNumThreadsOrCPUs();
  return 0;
}
