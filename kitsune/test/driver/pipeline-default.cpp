// If the --tapir=nolo option is provided without optimizations, neither tapir,
// nor Kitsune, passes are run.
//
// RUN: %kitxx --tapir=nolo -O0 -c -emit-llvm -o /dev/null %s \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s -check-prefix O0
//
// O0:      Running pass:     AlwaysInlinerPass
// O0-NEXT: Running analysis: ProfileSummaryAnalysis
// O0-NEXT: Running pass:     CoroConditionalWrapper
// O0-NEXT: Running pass:     AnnotationRemarksPass
// O0-NEXT: Running analysis: TargetLibraryAnalysis
// O0-NEXT: Running pass:     VerifierPass
// O0-NEXT: Running pass:     BitcodeWriterPass
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
// O123SZ:      Running pass:     AnnotateTapirLoopsPass
// O123SZ:      Running pass:     LoopSpawningPass
//
// FIXME: Remove the comment below once the loop strip-mining pass has been
// reverted to the original and George's reduction modifications have been moved
// into its own pass
// COM: O123SZ-NEXT: Running analysis: TapirTargetAnalysis
// COM: O123SZ-NEXT: Running analysis: TaskAnalysis
//
// O123SZ:      Running pass:     TapirToTargetPass
// O123SZ:      Running pass:     GlobalDCEPass
// O123SZ-NEXT: Running pass:     PrefetchingPass
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
