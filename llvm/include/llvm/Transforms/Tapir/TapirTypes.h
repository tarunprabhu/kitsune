//===- TapirTypes.h - Tapir types ------------------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file enumerates the available Tapir lowering targets.
//
//===----------------------------------------------------------------------===//

#ifndef TAPIR_TYPES_H_
#define TAPIR_TYPES_H_

namespace llvm {

enum class TapirTargetType {
  None = 0,
  Serial = 1,
  Cilk = 2,
  OpenMP = 3,
  CilkR = 4,
  Qthreads = 5,
  Cuda = 6,
  Realm = 7,
  Last_TapirTargetType = 8
};

} // end namespace llvm

#endif
