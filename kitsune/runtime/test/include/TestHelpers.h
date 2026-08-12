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
  const struct KitRTInitOptions initOpts = {RTS};                              \
                                                                               \
  __attribute__((constructor)) static void ctor(void) {                        \
    __kitrt_initialize(&initOpts);                                             \
  }                                                                            \
                                                                               \
  __attribute__((destructor)) static void dtor(void) {                         \
    __kitrt_finalize(&initOpts);                                               \
  }

#define MAIN                                                                   \
  int main(int argc, char *argv[]) { return 0; }

#define BOOLSTR(v) ((v) ? "true" : "false")

#endif // KITRT_UNITTESTS_TEST_HELPERS_H
