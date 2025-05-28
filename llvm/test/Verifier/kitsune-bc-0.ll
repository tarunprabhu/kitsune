; The verifier should not complain if there are is no embedded bitcode
;
; RUN: llvm-as %s -o /dev/null 

@0 = constant [0 x i8] zeroinitializer
