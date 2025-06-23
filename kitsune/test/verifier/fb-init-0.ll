; The initializer of global variables with the kit_fb attribute can be zero
;
; RUN: llvm-as %s -o /dev/null

@fb = constant [1 x i8] zeroinitializer #0

attributes #0 = { kit_fb(1) }
