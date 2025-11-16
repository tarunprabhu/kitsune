; The tapir-lowering meta-pass requires --tapir
;
; RUN: not opt -passes='tapir-lowering<O1>' -disable-output %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: tapir-lowering passes require the --tapir option
