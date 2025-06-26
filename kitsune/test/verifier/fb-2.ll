; Currently, only a single fat binary object is allowed for a given tapir
; target.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: too many embedded fat binary globals for tapir target 'cuda'

@bc.2 = constant [0 x i8] zeroinitializer #0
@.bc.2 = constant [0 x i8] zeroinitializer #0
@bc.4 = constant [0 x i8] zeroinitializer #1

attributes #0 = { kit_fb kit_tt(2) }
attributes #1 = { kit_fb kit_tt(4) }
