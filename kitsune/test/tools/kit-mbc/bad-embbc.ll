; If a global variable containing embedded bitcode was found, but that module
; could not be parsed, fail with an error
;
; RUN: BADBC="@.bc.bad = constant [2 x i8] c\"BC\" #2"
; RUN: ATTRS="attributes #2 = { kit_bc kit_tt(2) }"
;
; RUN: printf "${BADBC}\n${ATTRS}" \
; RUN:     | not %if asserts %{ --crash %} %kit-mbc -S -o - 2>&1 \
; RUN:     | FileCheck %s --check-prefix ERROR
;
; ------------------------------------------------------------------------------
;
; If a global variable contains embedded bitcode which cannot be parsed into an
; LLVM module, but that module is never requested, it is not an error
;
; RUN: echo "@hip = global i32 4" \
; RUN:     | %kit-enc --tapir=hip \
; RUN:     | sed -E "/^[@][.]kitsune[.]emb[.]fb/a\ ${BADBC}" \
; RUN:     | sed -E "/^attributes #1/a\ ${ATTRS}" \
; RUN:     | %kit-mbc --tapir=hip -S -o - 2>&1 \
; RUN:     | FileCheck %s --check-prefix HIP
;
; ------------------------------------------------------------------------------
;
; ERROR: error:
; HIP: @hip
