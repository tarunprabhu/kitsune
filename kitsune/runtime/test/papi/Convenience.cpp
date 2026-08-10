// REQUIRES: kitsune-papi
//
// Check the Kitsune-specific convenience names for events. New convenience
// names should be added to this test.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// If the event is available, a log message will be printed, otherwise, a
// warning message will be printed. In either case, the message will start with
// Event '<event-name>'
//
// CHECK: Event 'PAPI_L1_DCM'
// CHECK: Event 'PAPI_L2_DCM'
// CHECK: Event 'PAPI_L3_DCM'
// CHECK: Event 'PAPI_L1_ICM'
// CHECK: Event 'PAPI_L2_ICM'
// CHECK: Event 'PAPI_L3_ICM'
// CHECK: Event 'PAPI_L1_TCM'
// CHECK: Event 'PAPI_L2_TCM'
// CHECK: Event 'PAPI_L3_TCM'
// CHECK: Event 'PAPI_L1_LDM'
// CHECK: Event 'PAPI_L2_LDM'
// CHECK: Event 'PAPI_L3_LDM'
// CHECK: Event 'PAPI_L1_STM'
// CHECK: Event 'PAPI_L2_STM'
// CHECK: Event 'PAPI_L3_STM'
// CHECK: Event 'PAPI_TLB_DM'
// CHECK: Event 'PAPI_TLB_IM'
// CHECK: Event 'PAPI_TLB_TL'
// CHECK: Event 'PAPI_TOT_INS'
// CHECK: Event 'PAPI_TOT_INS'
// CHECK: Event 'PAPI_VEC_INS'
// CHECK: Event 'PAPI_LD_INS'
// CHECK: Event 'PAPI_SR_INS'
// CHECK: Event 'PAPI_BR_INS'
// CHECK: Event 'PAPI_INT_INS'
// CHECK: Event 'PAPI_FP_INS'
// CHECK: Event 'PAPI_FMA_INS'
// CHECK: Event 'PAPI_RES_STL'
// CHECK: Event 'PAPI_TOT_CYC'
// CHECK: Event 'PAPI_REF_CYC'

#include "TestHelpers.h"
#include "papi/kitpapi.h"

CTOR(RT_PAPI)

int main(int argc, char *argv[]) {
  kitrt::KitPAPIEpoch *e = __kitpapi_start(
      "mallika", /*thread=*/0, 30, "l1d", "l2d", "l3d", "l1i", "l2i", "l3i",
      "l1t", "l2t", "l3t", "l1ld", "l2ld", "l3ld", "l1st", "l2st", "l3st",
      "tlbd", "tlbi", "tlbt", "ins", "inst", "vec", "ld", "st", "br", "int",
      "fp", "fma", "stall", "cyc", "ref");
  __kitpapi_stop(e);

  return 0;
}
