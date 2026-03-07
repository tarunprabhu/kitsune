// If the --tapir=nolo option is provided without optimizations, neither tapir,
// nor Kitsune, passes are run.
//
// RUN: %kitxx --tapir=nolo -O0 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O0
//
// O0-NOT: Running pass: AnnotateTapirLoopsPass
// O0-NOT: Running pass: LoopSpawningPass
//
// -----------------------------------------------------------------------------
// If the --tapir argument is provided, all Tapir and Kitsune passes should run.
//
// RUN: %kitxx --tapir=serial -O1 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123SZ
//
// RUN: %kitxx --tapir=serial -O2 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123SZ
//
// RUN: %kitxx --tapir=serial -O3 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123SZ
//
// RUN: %kitxx --tapir=serial -Os -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123SZ
//
// RUN: %kitxx --tapir=serial -Oz -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O123SZ
//
// O123SZ:      Running pass:     AnnotationRemarksPass
// O123SZ-NEXT: Running pass:     LoopSimplifyPass
// O123SZ-NEXT: Running analysis: LoopAnalysis
// O123SZ-NEXT: Running analysis: DominatorTreeAnalysis
// O123SZ:      Running pass:     PreLowerVerificationPass
// O123SZ-NEXT: Running analysis: TapirTargetAnalysis
// O123SZ-NEXT: Running analysis: TaskAnalysis
// O123SZ-NEXT: Running analysis: PostDominatorTreeAnalysis
// O123SZ-NEXT: Running analysis: ScalarEvolutionAnalysis
// O123SZ-NEXT: Running pass:     AnnotateTapirLoopsPass
// O123SZ-NEXT: Running pass:     SerializePass
// O123SZ-NEXT: Running pass:     LoopSpawningPass
// O123SZ:      Running pass:     TapirToTargetPass
// O123SZ:      Running pass:     GlobalDCEPass
// O123SZ-NEXT: Running pass:     PrefetchForDevicePass
// O123SZ-NEXT: Running pass:     EmbResolveLibDeviceCallsPass
// O123SZ-NEXT: Running pass:     EmbPreparePass
// O123SZ-NEXT: Running pass:     EmbLinkLibDeviceBitcodePass
// O123SZ-NEXT: Running pass:     EmbOptimizePass
// O123SZ-NEXT: Running pass:     RecomputeKernelPropertiesPass
// O123SZ-NEXT: Running pass:     GenerateCtorsPass
// O123SZ-NEXT: Running pass:     VerifierPass
// O123SZ-NEXT: Running analysis: VerifierAnalysis
// O123SZ-NEXT: Running pass:     BitcodeWriterPass

void f() {}
