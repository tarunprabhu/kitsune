// If both kit++.cfg and clang++.cfg are present in the same directory, ensure
// that the correct configuration file is read.
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN:     not %kitxx --config-system-dir=%S/input/cfgs %s 2>&1 \
// RUN:         | FileCheck %s --check-prefixes=KITXX
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN:     not %kitxx --config-user-dir=%S/input/cfgs %s 2>&1 \
// RUN:         | FileCheck %s --check-prefixes=KITXX
//
// KITXX: error: unknown argument: '--not-a-kitxx-option'
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN:     not %clangxx --config-system-dir=%S/input/cfgs %s 2>&1 \
// RUN:         | FileCheck %s --check-prefixes=CLANGXX
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN:     not %clangxx --config-user-dir=%S/input/cfgs %s 2>&1 \
// RUN:         | FileCheck %s --check-prefixes=CLANGXX
//
// CLANGXX: error: unknown argument: '--not-a-clangxx-option'
