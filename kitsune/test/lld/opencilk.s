;;; REQUIRES: kitsune-opencilk

;;; RUN: not ld.lld --tapir=opencilk --tapir-opencilk-abi-bc= 2>&1 \
;;; RUN:     | FileCheck %s -check-prefix MISSING

;;; We cannot validate the argument to --tapir-opencilk-abi-bc, so all options
;;; here should be recognized and the error will be about missing inputs
;;;
;;; RUN: not ld.lld --tapir=opencilk --tapir-opencilk-abi-bc=/path/to/abi.bc \
;;; RUN:     2>&1 | FileCheck %s -check-prefix INPUTS

;;; MISSING: error: {{.+}}: missing argument
;;; INPUTS: error: no input files
