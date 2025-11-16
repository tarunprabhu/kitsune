; -----------------------------------------------------------------------------
; If the --tapir option is not provided to llc, the Kitsune passes are not run.
;
; RUN: llc -O0 -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: llc -O1 -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: llc -O2 -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; DEFAULT-NOT: Strip Kitsune address spaces
; DEFAULT-NOT: Lower Kitsune intrinsics
; DEFAULT-NOT: Generate Kitsune fat binaries
;
; -----------------------------------------------------------------------------
; If the --tapir option is provided to llc, the Kitsune passes are run at all
; optimization levels.
;
; RUN: llc -O0 --tapir=nolo -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix TAPIR %s
;
; RUN: llc -O1 --tapir=nolo -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix TAPIR %s
;
; RUN: llc -O2 --tapir=nolo -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix TAPIR %s
;
; RUN: llc -O3 --tapir=nolo -debug-pass=Structure %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix TAPIR %s
;
; TAPIR: Target Library Information
; TAPIR: ModulePass Manager
; TAPIR-NEXT: Lower Kitsune intrinsics
; TAPIR-NEXT: Strip Kitsune address spaces
; TAPIR-NEXT: Generate Kitsune fat binaries
;
; -----------------------------------------------------------------------------
