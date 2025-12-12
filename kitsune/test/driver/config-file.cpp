// Kitsune-specific config files
//
//------------------------------------------------------------------------------
// RUN: %kitxx --config-kitsune-dir=%S/inputs/config3 -o /dev/null -v 2>&1 \
// RUN:     | FileCheck %s -check-prefix DIR
// RUN: %kitxx --config-kitsune-dir=%S/inputs/config3 -o /dev/null -v 2>&1 \
// RUN:     | FileCheck %s -check-prefix DIR
//
// DIR: Kitsune configuration file directory: {{.*}}/inputs/config3
//
// -----------------------------------------------------------------------------
// The --config-kitsune-dir option can only be used with a Kitsune frontend.
//
// RUN: not %clang++ --config-kitsune-dir=%S/inputs/config3 -v 2>&1 \
// RUN:     | FileCheck %s -check-prefix FRONTEND
//
// FRONTEND: option '--config-kitsune-dir=' must be used with a Kitsune frontend
//
// -----------------------------------------------------------------------------
// Check that the kitsune config directory is examined for config files. The
// kitsune config directory is searched before --config-user-dir.
//
// RUN: not %kitxx --config nonexistent-config-file.cfg \
// RUN:     --config-system-dir=%S/inputs/config1 \
// RUN:     --config-user-dir=%S/inputs/config2 \
// RUN:     --config-kitsune-dir=%S/inputs/config3 2>&1 \
// RUN:     | FileCheck %s -check-prefix NOTFOUND
//
// NOTFOUND: configuration file 'nonexistent-config-file.cfg' cannot be found
// NOTFOUND-NEXT: was searched for in the directory: {{.*}}/inputs/config2
// NOTFOUND-NEXT: was searched for in the directory: {{.*}}/inputs/config3
// NOTFOUND-NEXT: was searched for in the directory: {{.*}}/inputs/config1
// NOTFOUND-NEXT: was searched for in the directory:
//
// -----------------------------------------------------------------------------
// If both kit++.cfg and clang++.cfg are present in the same directory, ensure
// that the correct configuration file is picked up.
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN:     not %kitxx --config-system-dir=%S/input/cfgs %s 2>&1 \
// RUN:         | FileCheck %s --check-prefixes=KITXX
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN:     not %kitxx --config-user-dir=%S/input/cfgs %s 2>&1 \
// RUN:         | FileCheck %s --check-prefixes=KITXX
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN:     not %kitxx --config-kitsune-dir=%S/input/cfgs %s 2>&1 \
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
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN:     not %clangxx --config-kitsune-dir=%S/input/cfgs %s 2>&1 \
// RUN:         | FileCheck %s --check-prefixes=CLANGXX
//
// CLANGXX: error: unknown argument: '--not-a-clangxx-option'
