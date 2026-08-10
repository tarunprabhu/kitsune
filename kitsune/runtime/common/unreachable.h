//===- Unreachable.h - Utilities for catastrophic errors --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for catastrophic errors.
//
//===----------------------------------------------------------------------===//

#ifndef KITRT_COMMON_UNREACHABLE_H
#define KITRT_COMMON_UNREACHABLE_H

#include <cstdio>
#include <cstdlib>

#define UNREACHABLE(msg)                                                       \
  do {                                                                         \
    if (msg)                                                                   \
      fprintf(stderr, "%s\n", msg);                                            \
    fprintf(stderr, "UNREACHABLE executed");                                   \
    if (__FILE__)                                                              \
      fprintf(stderr, " at %s:%d\n", __FILE__, __LINE__);                      \
    fprintf(stderr, "\n");                                                     \
    abort();                                                                   \
  } while (0)

#endif // KITRT_COMMON_UNREACHABLE_H
