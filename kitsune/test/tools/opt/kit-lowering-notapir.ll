; The kit-lowering meta-pass requires --tapir
;
; RUN: not opt -passes='kit-lowering<O1>' -disable-output %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: kit-lowering passes require the --tapir option
