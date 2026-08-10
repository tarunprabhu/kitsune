// REQUIRES: kitsune-papi, amd-gpu
//
// PAPI can be used when the hip tapir target is enabled. In this case, a thread
// ID function will not be used.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initializing Kitsune runtime (hip)
// CHECK: Initialized Kitsune runtime (hip)
// CHECK: Initializing Kitsune runtime (papi)
// CHECK-NOT: Initializing PAPI threading support
// CHECK-NOT: Initialized PAPI threading support
// CHECK: Initialized Kitsune runtime (papi)

#include "TestHelpers.h"

CTOR(RT_PAPI | RT_HIP)

int main(int argc, char *argv[]) { return 0; }
