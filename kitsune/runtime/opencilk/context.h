//===- opencilk/context.h - Context for the opencilk runtime ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Context object for any state required by the Kitsune's opencilk runtime.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_OPENCILK_CONTEXT_H
#define KITRT_OPENCILK_CONTEXT_H

#include "common/thread.h"

#include <cstdint>

namespace kitrt {

/// Kitsune runtime the opencilk tapir target. All global state required by the
/// runtime should be owned by this object.
class OpenCilkContext {
public:
  void initialize();
  void finalize();
  uint64_t getNumThreads() const;
  KitThreadID getThreadID() const;
};

} // namespace kitrt

#endif // KITRT_OPENCILK_CONTEXT_H
