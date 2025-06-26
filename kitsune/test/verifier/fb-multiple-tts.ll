; Multiple fat binaries are allowed in a host module as long as they are from
; distinct tapir targets.
;
; RUN: llvm-as %s -o /dev/null

@bc.2 = constant [0 x i8] zeroinitializer #0
@bc.4 = constant [0 x i8] zeroinitializer #1

attributes #0 = { kit_fb kit_tt(2) }
attributes #1 = { kit_fb kit_tt(4) }
