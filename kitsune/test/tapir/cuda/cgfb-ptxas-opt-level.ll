; Check that the -cgfb-ptxas-O<N> option is handled correctly.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda -disable-output \
; RUN:           -passes='kit-cgfb' -cgfb-### 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,DEFAULT
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda -disable-output \
; RUN:           -passes='kit-cgfb' -cgfb-### -cgfb-ptxas-O0 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,O0
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda -disable-output \
; RUN:           -passes='kit-cgfb' -cgfb-### -cgfb-ptxas-O1 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,O1
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda -disable-output \
; RUN:           -passes='kit-cgfb' -cgfb-### -cgfb-ptxas-O2 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,O2
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda -disable-output \
; RUN:           -passes='kit-cgfb' -cgfb-### -cgfb-ptxas-O3 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,O3
;
; RUN: not opt --tapir=cuda %s -disable-output \
; RUN:         -passes='kit-cgfb' -cgfb-### -cgfb-ptxas-Os %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes OS
;
; RUN: not opt --tapir=cuda %s -disable-ouput \
; RUN:         -passes='kit-cgfb' -cgfb-### -cgfb-ptxas-Oz 2>&1 \
; RUN:     | FileCheck %s --check-prefixes OZ
;
; ALL: ptxas
; DEFAULT-SAME: --opt-level 1
; O0-SAME: --opt-level 0
; O1-SAME: --opt-level 1
; O2-SAME: --opt-level 2
; O3-SAME: --opt-level 3
; OS: Unknown command line argument '-cgfb-ptxas-Os'
; OZ: Unknown command line argument '-cgfb-ptxas-Oz'

