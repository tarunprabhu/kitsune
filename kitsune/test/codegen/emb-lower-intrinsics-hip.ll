; REQUIRES: kitsune-hip
;
; At this time, all hip intrinsics are lowered earlier in the pass pipeline,
; the pass will fail if calls to these intrinsics are present. But it is not
; worth checking for that since it is a catastrophic failure. If we ever change
; it to being a more graceful error, we might add tests here. Until then, we
; keep this empty placeholder for symmetry with cuda, but ensure that it always
; passes.
;
; RUN: true
