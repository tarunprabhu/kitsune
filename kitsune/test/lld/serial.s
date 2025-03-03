;;; RUN: not ld.lld --tapir=bad 2>&1 | FileCheck %s -check-prefix TT
;;; RUN: not ld.lld --tapir= 2>&1 | FileCheck %s -check-prefix TT

;;; If these flags are present and recognized, lld should complain about the
;;; lack of any inputs
;;;
;;; RUN: not ld.lld --kitrt-verbose 2>&1 | FileCheck %s -check-prefix NO-FILES
;;; RUN: not ld.lld --tapir-verbose 2>&1 | FileCheck %s -check-prefix NO-FILES

;;; TT: invalid value '{{.*}}' in '--tapir'
;;; NO-FILES: error: no input files
