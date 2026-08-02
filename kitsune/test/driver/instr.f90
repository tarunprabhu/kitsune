! REQUIRES: kitfc
!
! Check Kitsune's instrumentation related options.
!
! ------------------------------------------------------------------------------
! --kit-instr= valid
!
! RUN: %kitxx -### %s 2>&1 | FileCheck %s --check-prefixes=CC1,KIND-DEFAULT
!
! RUN: %kitxx -### --kit-instr=generic %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,KIND-GENERIC
!
! RUN: %kitxx -### --kit-instr=papi %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,KIND-PAPI
!
! RUN: %kitxx -### --kit-instr=timer %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,KIND-TIMER
!
! RUN: %kitxx -### --kit-instr=generic,papi %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,KIND-MULTIPLE2
!
! RUN: %kitxx -### --kit-instr=timer,generic,papi %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,KIND-MULTIPLE3
!
! RUN: %kitxx -### --kit-instr=papi,generic,papi %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,KIND-REPEAT
!
! CC1: "-cc1"
!
! KIND-DEFAULT-NOT: --kit-instr=
!
! KIND-GENERIC-SAME: "--kit-instr=generic"
! KIND-PAPI-SAME: "--kit-instr=papi"
! KIND-TIMER-SAME: "--kit-instr=timer"
! KIND-MULTIPLE2-SAME: "--kit-instr=generic,papi"
! KIND-MULTIPLE3-SAME: "--kit-instr=timer,generic,papi"
! KIND-REPEAT-SAME: "--kit-instr=papi,generic,papi"
!
! ------------------------------------------------------------------------------
! --kit-instr= invalid
!
! RUN: not %kitxx -### --kit-instr= %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=EMPTY-LIST
!
! RUN: not %kitxx -### --kit-instr=timers %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=KIND-INVALID-TIMERS
!
! RUN: not %kitxx -### --kit-instr=PAPI %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=KIND-INVALID-PAPI
!
! RUN: not %kitxx -### --kit-instr=generic,counters %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=KIND-INVALID-COUNTERS
!
! EMPTY-LIST: argument to '{{.+}}' must be comma-separated list with at least one element
! KIND-INVALID-TIMERS: invalid value 'timers' in '{{.+}}'
! KIND-INVALID-PAPI: invalid value 'PAPI' in '{{.+}}'
! KIND-INVALID-COUNTERS: invalid value 'counters' in '{{.+}}'
!
! ------------------------------------------------------------------------------
! --kit-instr-only= valid
!
! RUN: %kitxx -### %s 2>&1 | FileCheck %s --check-prefixes=CC1,ONLY-DEFAULT
!
! RUN: %kitxx -### --kit-instr-only=one %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,ONLY-ONE
!
! RUN: %kitxx -### --kit-instr-only=one,two,three %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,ONLY-THREE
!
! ONLY-DEFAULT-NOT: --kit-instr-only
! ONLY-ONE-SAME: "--kit-instr-only=one"
! ONLY-THREE-SAME: "--kit-instr-only=one,two,three"
!
! ------------------------------------------------------------------------------
! --kit-instr-only= invalid
!
! RUN: not %kitxx -### --kit-instr-only= %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=EMPTY-LIST
!
! RUN: not %kitxx -### --kit-instr-only="this,,that" %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=ONLY-INVALID-EMPTY
!
! ONLY-INVALID-EMPTY: invalid value '' in '{{.+}}'
!
! ------------------------------------------------------------------------------
! --kit-instr-unit= valid
!
! RUN: %kitxx -### %s 2>&1 | FileCheck %s --check-prefixes=CC1,UNIT-DEFAULT
!
! RUN: %kitxx -### --kit-instr-unit=all %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,UNIT-ALL
!
! RUN: %kitxx -### --kit-instr-unit=default %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,UNIT-DEFAULTS
!
! RUN: %kitxx -### --kit-instr-unit=loop %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,UNIT-LOOP
!
! RUN: %kitxx -### --kit-instr-unit=thread %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,UNIT-THREAD
!
! RUN: %kitxx -### --kit-instr-unit=loop,thread %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,UNIT-MULTIPLE2
!
! RUN: %kitxx -### --kit-instr-unit=loop,thread,loop %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,UNIT-REPEAT
!
! UNIT-DEFAULT-NOT: --kit-instr-unit=
! UNIT-ALL-SAME: "--kit-instr-unit=all"
! UNIT-DEFAULTS-SAME: "--kit-instr-unit=default"
! UNIT-LOOP-SAME: "--kit-instr-unit=loop"
! UNIT-THREAD-SAME: "--kit-instr-unit=thread"
! UNIT-MULTIPLE2-SAME: "--kit-instr-unit=loop,thread"
! UNIT-REPEAT-SAME: "--kit-instr-unit=loop,thread,loop"
!
! ------------------------------------------------------------------------------
! --kit-instr-unit= invalid
!
! RUN: not %kitxx -### --kit-instr-unit= 2>&1 \
! RUN:     | FileCheck %s --check-prefix=EMPTY-LIST
!
! RUN: not %kitxx -### --kit-instr-unit=ALL 2>&1 \
! RUN:     | FileCheck %s --check-prefix=UNIT-INVALID-ALL
!
! RUN: not %kitxx -### --kit-instr-unit=loops 2>&1 \
! RUN:     | FileCheck %s --check-prefix=UNIT-INVALID-LOOPS
!
! RUN: not %kitxx -### --kit-instr-unit=loop,function 2>&1 \
! RUN:     | FileCheck %s --check-prefix=UNIT-INVALID-FUNCTION
!
! RUN: not %kitxx -### --kit-instr-unit=all,loop 2>&1 \
! RUN:     | FileCheck %s --check-prefix=UNIT-ALL-LIST
!
! RUN: not %kitxx -### --kit-instr-unit=default,loop 2>&1 \
! RUN:     | FileCheck %s --check-prefix=UNIT-DEFAULT-LIST
!
! UNIT-INVALID-ALL: invalid value 'ALL' in '{{.+}}'
! UNIT-INVALID-LOOPS: invalid value 'loops' in '{{.+}}'
! UNIT-INVALID-FUNCTION: invalid value 'function' in '{{.+}}'
! UNIT-ALL-LIST: 'all' cannot appear in list in '--kit-instr-unit={{.+}}'
! UNIT-DEFAULT-LIST: 'default' cannot appear in list in '--kit-instr-unit={{.+}}'
!
! -----------------------------------------------------------------------------
! --kit-instr-papi= valid
!
! RUN: %kitfc -### %s 2>&1 | FileCheck %s --check-prefixes=CC1,PAPI-DEFAULT
!
! RUN: %kitfc -### --kit-instr-papi=vec %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,PAPI-ONE
!
! RUN: %kitfc -### --kit-instr-papi=tlbt,ca_itv,three %s 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=CC1,PAPI-THREE
!
! PAPI-DEFAULT-NOT: --kit-instr-papi
! PAPI-ONE-SAME: "--kit-instr-papi=vec"
! PAPI-THREE-SAME: "--kit-instr-papi=tlbt,ca_itv,three"
!
! -----------------------------------------------------------------------------
! --kit-instr-papi= invalid
!
! RUN: not %kitfc -### --kit-instr-papi= %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=EMPTY-LIST
!
! RUN: not %kitfc -### --kit-instr-papi="fma,,l1_icw" %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=PAPI-INVALID-EMPTY
!
! RUN: not %kitfc -fc1 -fsyntax-only --kit-instr=papi %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=PAPI-MISSING
!
! PAPI-INVALID-EMPTY: invalid value '' in '{{.+}}'
! PAPI-MISSING: missing required option
!
! -----------------------------------------------------------------------------
