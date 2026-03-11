; Check that opt's command line options make it to the tapir target options.
;
; RUN: opt --tapir=hip \
; RUN:     --tapir-gpu-tpb=64 \
; RUN:     --tapir-gpu-max-tpb=128 \
; RUN:     --tapir-gpu-prefetch=false \
; RUN:     --tapir-hip-arch=gfx906 \
; RUN:     --tapir-hip-sramecc=off \
; RUN:     --tapir-hip-xnack=on \
; RUN:     --tapir-hip-features="-sramecc,+xnack" \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     --tapir-lld="%S/input/ld.lld" 2>&1 \
; RUN:     -dump-tapir-target-options \
; RUN:     -passes="loop-spawning" -o /dev/null %s \
; RUN:     | FileCheck %s -check-prefixes ALL,CHECK
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90a \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     --tapir-gpu-prefetch=true 2>&1 \
; RUN:     -dump-tapir-target-options \
; RUN:     -passes="loop-spawning" -o /dev/null %s \
; RUN:     | FileCheck %s -check-prefixes ALL,PREFETCH
;
; ALL:       Tapir target options
; ALL:       Primary: hip
; CHECK:     GPU fixed threads/block: 64
; CHECK:     GPU max threads/block: 128
; CHECK:     GPU prefetch: 0
; CHECK:     Hip arch: gfx906
; CHECK:     Hip sramecc: off
; CHECK:     Hip xnack: on
; CHECK:     Hip target features: -sramecc,+xnack
; CHECK:     Hip bitcode files: [
; CHECK:       {{.+}}/input/amd.bc
; CHECK:     ]
; PREFETCH:  GPU prefetch: 1
