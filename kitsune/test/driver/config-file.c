// If both kitcc.cfg and clang.cfg are present in the same directory, ensure
// that the correct configuration file is read.
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN:     not %kitcc --config-system-dir=%S/input/cfgs %s 2>&1 \
// RUN:         | FileCheck %s --check-prefixes=KITCC
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN:     not %kitcc --config-user-dir=%S/input/cfgs %s 2>&1 \
// RUN:         | FileCheck %s --check-prefixes=KITCC
//
// KITCC: error: unknown argument: '--not-a-kitcc-option'
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN:     not %clang --config-system-dir=%S/input/cfgs %s 2>&1 \
// RUN:         | FileCheck %s --check-prefixes=CLANG
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN:     not %clang --config-user-dir=%S/input/cfgs %s 2>&1 \
// RUN:         | FileCheck %s --check-prefixes=CLANG
//
// CLANG: error: unknown argument: '--not-a-clang-option'
