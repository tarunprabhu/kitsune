! REQUIRES: kitfc
!
! Check that Kitsune-specific instrumentation related options make it to the
! LLVM passes as expected. fc1 does not carry out any error checking, so all
! options must have valid values.
!
! RUN: %kitfc -fc1 --tapir=serial -O1 -emit-llvm -o /dev/null %s \
! RUN:     -mllvm --kit-instr-dump-opts \
! RUN:     --kit-instr=generic \
! RUN:     | FileCheck %s --check-prefixes=ALL,KINDG,UNIT-DEFAULT,NAMES-DEFAULT
!
! RUN: %kitfc -fc1 --tapir=serial -O1 -emit-llvm -o /dev/null %s \
! RUN:     -mllvm --kit-instr-dump-opts \
! RUN:     --kit-instr=timer,papi --kit-instr-unit=all \
! RUN:     | FileCheck %s --check-prefixes=ALL,KIND2,UNIT-ALL,NAMES-DEFAULT
!
! RUN: %kitfc -fc1 --tapir=serial -O1 -emit-llvm -o /dev/null %s \
! RUN:     -mllvm --kit-instr-dump-opts \
! RUN:     --kit-instr=timer --kit-instr-unit=default \
! RUN:     | FileCheck %s --check-prefixes=ALL,KINDT,UNIT-DEFAULT,NAMES-DEFAULT
!
! RUN: %kitfc -fc1 --tapir=serial -O1 -emit-llvm -o /dev/null %s \
! RUN:     -mllvm --kit-instr-dump-opts \
! RUN:     --kit-instr=papi --kit-instr-unit=thread --kit-instr-only="this,that" \
! RUN:     | FileCheck %s --check-prefixes=ALL,KINDP,UNITT,NAMES2
!
! ALL: Kitsune instrumentation options
!
! KINDG-NEXT: Kinds: generic
! KINDP-NEXT: Kinds: papi
! KINDT-NEXT: Kinds: timer
! KIND2-NEXT: Kinds: papi,timer
!
! UNIT-ALL-NEXT:     Units: thread,loop
! UNIT-DEFAULT-NEXT: Units: loop
! UNITT-NEXT:        Units: thread
!
! NAMES-DEFAULT-NEXT: Only: {{$}}
! NAMES2-NEXT:        Only: that,this

subroutine f()
end subroutine f
