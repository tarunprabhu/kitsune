// If the --tapir argument is not given, none of Kitsune's passes should run.
//
// RUN: %kitxx -O0 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
//
// RUN: %kitxx -O1 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
//
// RUN: %kitxx -O2 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
//
// RUN: %kitxx -O3 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
//
// RUN: %kitxx -Os -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
//
// RUN: %kitxx -Oz -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK-NOT:  Running pass:     EarlyAnnotatePass
// CHECK-NOT:  Running analysis: TTObjectsAnalysis
// CHECK-NOT:  Running pass:     PrefetchingPass
// CHECK-NOT:  Running pass:     EmbResolveLibDeviceCallsPass
// CHECK-NOT:  Running pass:     EmbPreparePass
// CHECK-NOT:  Running pass:     EmbLinkLibDeviceBitcodePass
// CHECK-NOT:  Running pass:     EmbOptimizePass
// CHECK-NOT:  Running pass:     RecomputeKernelPropertiesPass
// CHECK-NOT:  Running pass:     GenerateCtorsPass
