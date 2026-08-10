// REQUIRES: kitsune-papi
//
// The openmp tapir target should be initialized before PAPI.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initializing Kitsune runtime (openmp)
// CHECK: Initialized Kitsune runtime (openmp)
// CHECK: Initializing Kitsune runtime (papi)
// CHECK: Initializing PAPI threading support
// CHECK: Initialized PAPI threading support
// CHECK: Initialized Kitsune runtime (papi)

#include "TestHelpers.h"

CTOR(RT_PAPI | RT_OPENMP)

int main(int argc, char *argv[]) { return 0; }
