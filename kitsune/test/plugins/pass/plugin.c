// REQUIRES: kitsune-examples
//
// Check that the passes in a pass plugin are registered at the correct places
// in the pass pipeline.
//
// RUN: %kitcc --tapir=serial -O1 -S -emit-llvm -o /dev/null %s \
// RUN:     -fpass-plugin=%kit-pass-plugin-demo \
// RUN:     -Xclang -fdebug-pass-manager 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK: PreTapirEarlyPass
// CHECK: PreTapirLatePass
// CHECK: LoopSpawningPass
// CHECK: PostTapirEarlyPass
// CHECK: PostTapirLatePass
// CHECK: GenerateCtorsPass
// CHECK: PostTapirLastPass
