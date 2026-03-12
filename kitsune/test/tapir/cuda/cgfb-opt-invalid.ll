; Check that invalid cgfb optimization levels are handled correctly.
;
; RUN: not opt --tapir=cuda -passes='kit-cgfb' -disable-output %s \
; RUN:     -cgfb-Os 2>&1 \
; RUN:     | FileCheck %s --check-prefix OS
;
; RUN: not opt --tapir=cuda -passes='kit-cgfb' -disable-output %s \
; RUN:     -cgfb-Oz 2>&1 \
; RUN:     | FileCheck %s --check-prefix OZ
;
; RUN: not opt --tapir=cuda -passes='kit-cgfb' -disable-output %s \
; RUN:     -cgfb-O4 2>&1 \
; RUN:     | FileCheck %s --check-prefix O4
;
; OS: Unknown command line argument '-cgfb-Os'
; OZ: Unknown command line argument '-cgfb-Oz'
; O4: Unknown command line argument '-cgfb-O4'
