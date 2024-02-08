// Kitsune-specific config files

//------------------------------------------------------------------------------
// RUN: %kitcc --config-kitsune-dir=%S/inputs/config3 -o /dev/null -v 2>&1 \
// RUN:     | FileCheck %s -check-prefix DIR
// RUN: %kitxx --config-kitsune-dir=%S/inputs/config3 -o /dev/null -v 2>&1 \
// RUN:     | FileCheck %s -check-prefix DIR
//
// DIR: Kitsune configuration file directory: {{.*}}/inputs/config3

// -----------------------------------------------------------------------------
// The --config-kitsune-dir option can only be used with a Kitsune frontend.
//
// RUN: not %clang --config-kitsune-dir=%S/inputs/config3 -o /dev/null -v 2>&1 \
// RUN:     | FileCheck %s -check-prefix FRONTEND
//
// FRONTEND: option '--config-kitsune-dir=' must be used with a Kitsune frontend

// -----------------------------------------------------------------------------
// Check that the kitsune config directory is examined for config files
//
// RUN: not %kitcc --config nonexistent-config-file.cfg \
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
