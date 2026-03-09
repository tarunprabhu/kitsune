; Check that the expected error messages are printed when the passes required
; by the kit-serialize pass are not run.
;
; RUN: not opt -passes='kit-serialize' -disable-output %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: 'SerializePass': required pass 'AnnotateTapirLoopsPass' has not been run
