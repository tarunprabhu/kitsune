//===- TestHelpers.h - Helpers for Kitsune's system tests -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper functions and macros for Kitsune's system tests.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_UNITTESTS_TEST_HELPERS_H
#define KITRT_UNITTESTS_TEST_HELPERS_H

#include "kitrt.h"
#include "kitsune/Shared/RTInitOptions.h"

#define CTOR(RTS)                                                              \
  const KitRTInitOptions initOpts{RTS};                                        \
                                                                               \
  __attribute__((constructor)) static void ctor() {                            \
    __kitrt_initialize(&initOpts);                                             \
  }                                                                            \
                                                                               \
  __attribute__((destructor)) static void dtor() {                             \
    __kitrt_finalize(&initOpts);                                               \
  }

#define BOOLSTR(v) ((v) ? "true" : "false")

#endif // KITRT_UNITTESTS_TEST_HELPERS_H
