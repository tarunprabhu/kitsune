;;; REQUIRES: kitsune-cuda

;;; RUN: not ld.lld --tapir=cuda -O 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix MISSING
;;; RUN: not ld.lld --tapir=cuda -Os 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix FORMAT
;;; RUN: not ld.lld --tapir=cuda -O-1 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix RANGE
;;; RUN: not ld.lld --tapir=cuda -O4 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix RANGE
;;;
;;; ---------------------------------------------------------------------------
;;;
;;; RUN: not ld.lld --tapir=cuda --fp-contract= 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=MISSING
;;; RUN: not ld.lld --tapir=cuda --fp-contract=Off 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=RANGE
;;; RUN: not ld.lld --tapir=cuda --fp-contract=On 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=RANGE
;;; RUN: not ld.lld --tapir=cuda --fp-contract=FAST 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=RANGE
;;; RUN: not ld.lld --tapir=cuda --fp-contract=fast-honor-pragmas 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=RANGE
;;;
;;; For valid values of --fp-contract, lld should complain about the lack of
;;; inputs
;;;
;;; RUN: not ld.lld --tapir=hip --fp-contract=off 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=INPUTS
;;; RUN: not ld.lld --tapir=hip --fp-contract=on 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=INPUTS
;;; RUN: not ld.lld --tapir=hip --fp-contract=fast 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=INPUTS
;;;
;;; ---------------------------------------------------------------------------
;;;
;;; RUN: not ld.lld --tapir=cuda --tapir-threads-per-block= 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=MISSING
;;; RUN: not ld.lld --tapir=cuda --tapir-threads-per-block=yes 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=FORMAT
;;; RUN: not ld.lld --tapir=cuda --tapir-threads-per-block=-1 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=RANGE
;;; RUN: not ld.lld --tapir=cuda --tapir-threads-per-block=0 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=RANGE
;;; RUN: not ld.lld --tapir=cuda --tapir-threads-per-block=1025 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=RANGE
;;;
;;; ---------------------------------------------------------------------------
;;;
;;; RUN: not ld.lld --tapir=cuda --tapir-max-threads-per-block= 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=MISSING
;;; RUN: not ld.lld --tapir=cuda --tapir-max-threads-per-block=yes 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=FORMAT
;;; RUN: not ld.lld --tapir=cuda --tapir-max-threads-per-block=-1 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=RANGE
;;; RUN: not ld.lld --tapir=cuda --tapir-max-threads-per-block=0 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=RANGE
;;;
;;; ---------------------------------------------------------------------------
;;;
;;; RUN: not ld.lld --tapir=cuda --tapir-cuda-arch= 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix=MISSING
;;;
;;; ---------------------------------------------------------------------------
;;;
;;; MISSING: error: {{.+}}: missing argument
;;; FORMAT: error: {{.+}}: number expected
;;; RANGE: error: invalid value '{{.*}}' in {{.+}}
;;; INPUTS: error: no input files
