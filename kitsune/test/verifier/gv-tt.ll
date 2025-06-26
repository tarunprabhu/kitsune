; It is ok to add the kit_tt attribute to a global that does not contain
; embedded bitcode, a fat binary or anything else. It may not be useful, but it
; is not wrong.
;
; RUN: llvm-as -o /dev/null %s

@0 = external global i32 #0

attributes #0 = { kit_tt(2) }
