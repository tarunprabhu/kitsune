// The default FP contract value is different from clang's defaults. Check that
// the defaults have not changed when running without a Kitsune frontend/
//
// RUN: %clang -### -x hip -nogpuinc -nogpulib %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix DEFAULT-HIP
//
// When compiling for hip, the host and device code are compiled separately
// The device code compilation (with a "primary" triple indicating nvptx) does
// not contain an ffp-contract entry - presumably because it is set to
// "fast-honor-pragmas" internally. On the host (with an "auxiliary" triple
// indicating amdgcn), the fp-contract value does not appear either. It is not
// clear from this test what it gets set to.
//
// DEFAULT-HIP: "-triple" "amd{{.*}}-amd-amdhsa"
// DEFAULT-HIP-NOT: -ffp-contract
// DEFAULT-HIP: "-aux-triple" "amd{{.*}}-amd-amdhsa"
// DEFAULT-HIP-NOT: "-ffp-contract=on"
