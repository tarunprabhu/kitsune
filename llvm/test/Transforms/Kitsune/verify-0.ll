; Check that the verifier pass does not fail when given a module without any
; embedded bitcode modules.
;
; RUN: opt --tapir=serial -passes=verify-emb-bc %s -o /dev/null

@g = global i32 0