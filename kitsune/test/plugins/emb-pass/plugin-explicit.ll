; REQUIRES: kitsune-examples
;
; Check that the embedded module passes in a pass plugin work correctly when
; explicitly specified in -passes.
;
; NOTE: In each case, --tapir must be passed to opt, without which the
; embedded bitcode passes will not run.
;
; RUN: %if kitsune-cuda %{ \
; RUN:   kit-enc --tapir=cuda %s \
; RUN:       | opt --tapir=cuda \
; RUN:             --load-pass-plugin=%kit-emb-pass-plugin-demo \
; RUN:             --passes='emb-func-names' -disable-output \
; RUN:       | FileCheck %s \
; RUN: %}
;
; RUN: %if kitsune-hip %{ \
; RUN:   kit-enc --tapir=hip %s \
; RUN:       | opt --tapir=hip \
; RUN:             --load-pass-plugin=%kit-emb-pass-plugin-demo \
; RUN:             --passes='emb-func-names' -disable-output \
; RUN:       | FileCheck %s \
; RUN: %}
;
; CHECK-DAG: device_f
; CHECK-DAG: device_g

define i64 @device_f(i64 %i) {
  ret i64 %i
}

define void @device_g(i64 %i) {
  ret void
}
