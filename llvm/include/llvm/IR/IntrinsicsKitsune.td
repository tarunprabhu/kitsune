//===- IntrinsicsKitsune.td - Defines Kitsune intrinsics ---*- tablegen -*-===//
//
// 
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the Kitsune-specific intrinsics.
//
//===----------------------------------------------------------------------===//

// 
def int_kitsune_launch_params:
  DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>],
  [IntrNoMem, IntrSpeculatable]
>;
