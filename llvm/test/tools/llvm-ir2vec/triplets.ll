; RUN: llvm-ir2vec triplets %s | FileCheck %s -check-prefix=TRIPLETS

define i32 @simple_add(i32 %a, i32 %b) {
entry:
  %add = add i32 %a, %b
  ret i32 %add
}

define i32 @simple_mul(i32 %x, i32 %y) {
entry:
  %mul = mul i32 %x, %y
  ret i32 %mul
}

define i32 @test_function(i32 %arg1, i32 %arg2) {
entry:
  %local1 = alloca i32, align 4
  %local2 = alloca i32, align 4
  store i32 %arg1, ptr %local1, align 4
  store i32 %arg2, ptr %local2, align 4
  %load1 = load i32, ptr %local1, align 4
  %load2 = load i32, ptr %local2, align 4
  %result = add i32 %load1, %load2
  ret i32 %result
}

; TRIPLETS: MAX_RELATION=3
; TRIPLETS-NEXT: 16	78	0
; TRIPLETS-NEXT: 16	88	2
; TRIPLETS-NEXT: 16	88	3
; TRIPLETS-NEXT: 16	0	1
; TRIPLETS-NEXT: 0	73	0
; TRIPLETS-NEXT: 0	88	2
; TRIPLETS-NEXT: 20	78	0
; TRIPLETS-NEXT: 20	88	2
; TRIPLETS-NEXT: 20	88	3
; TRIPLETS-NEXT: 20	0	1
; TRIPLETS-NEXT: 0	73	0
; TRIPLETS-NEXT: 0	88	2
; TRIPLETS-NEXT: 34	81	0
; TRIPLETS-NEXT: 34	87	2
; TRIPLETS-NEXT: 34	34	1
; TRIPLETS-NEXT: 34	81	0
; TRIPLETS-NEXT: 34	87	2
; TRIPLETS-NEXT: 34	36	1
; TRIPLETS-NEXT: 36	73	0
; TRIPLETS-NEXT: 36	88	2
; TRIPLETS-NEXT: 36	86	3
; TRIPLETS-NEXT: 36	36	1
; TRIPLETS-NEXT: 36	73	0
; TRIPLETS-NEXT: 36	88	2
; TRIPLETS-NEXT: 36	86	3
; TRIPLETS-NEXT: 36	35	1
; TRIPLETS-NEXT: 35	78	0
; TRIPLETS-NEXT: 35	86	2
; TRIPLETS-NEXT: 35	35	1
; TRIPLETS-NEXT: 35	78	0
; TRIPLETS-NEXT: 35	86	2
; TRIPLETS-NEXT: 35	16	1
; TRIPLETS-NEXT: 16	78	0
; TRIPLETS-NEXT: 16	88	2
; TRIPLETS-NEXT: 16	88	3
; TRIPLETS-NEXT: 16	0	1
; TRIPLETS-NEXT: 0	73	0
; TRIPLETS-NEXT: 0	88	2
