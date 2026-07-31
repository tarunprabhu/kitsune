; Check the command-line options in kit-instrument pass work as expected. These
; are intended to be used when working with the pass and opt directly.
;
; RUN: opt -passes=kit-instrument -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --allow-empty --check-prefixes=NOTHING
;
; RUN: opt -passes=kit-instrument --kit-instr-dump-opts -disable-output %s \
; RUN:     | FileCheck %s --check-prefixes=ALL,DISABLED
;
; RUN: opt -passes=kit-instrument --kit-instr-dump-opts -disable-output  %s \
; RUN:     --kit-instr=generic \
; RUN:     | FileCheck %s --check-prefixes=ALL,KINDG,UNIT-DEFAULT,NAMES-DEFAULT
;
; RUN: opt -passes=kit-instrument --kit-instr-dump-opts -disable-output  %s \
; RUN:     --kit-instr=timer,papi --kit-instr-unit=all \
; RUN:     | FileCheck %s --check-prefixes=ALL,KIND2,UNIT-ALL,NAMES-DEFAULT
;
; RUN: opt -passes=kit-instrument --kit-instr-dump-opts -disable-output  %s \
; RUN:     --kit-instr=timer --kit-instr-unit=default \
; RUN:     | FileCheck %s --check-prefixes=ALL,KINDT,UNIT-DEFAULT,NAMES-DEFAULT
;
; RUN: opt -passes=kit-instrument --kit-instr-dump-opts -disable-output  %s \
; RUN:     --kit-instr=papi --kit-instr-unit=thread --kit-instr-only="this,that" \
; RUN:     | FileCheck %s --check-prefixes=ALL,KINDP,UNITT,NAMES2
;
; NOTHING-NOT: {{.+}}
;
; ALL: Kitsune instrumentation options
;
; DISABLED-NEXT: Kinds:{{$}}
; DISABLED-NEXT: Units:{{$}}
; DISABLED-NEXT: Only:{{$}}
;
; KINDG-NEXT: Kinds: generic
; KINDP-NEXT: Kinds: papi
; KINDT-NEXT: Kinds: timer
; KIND2-NEXT: Kinds: papi,timer
;
; UNIT-ALL-NEXT:     Units: thread,loop
; UNIT-DEFAULT-NEXT: Units: loop
; UNITT-NEXT:        Units: thread
;
; NAMES-DEFAULT-NEXT: Only: {{$}}
; NAMES2-NEXT:        Only: that,this

define void @f() {
  ret void
}
