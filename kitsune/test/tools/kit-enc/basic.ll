; Check basic usage and command line options
;
; ------------------------------------------------------------------------------
; The default tapir target to add to the encoded module is cuda.
;
; RUN: %kit-enc %s \
; RUN:     | FileCheck %s --check-prefix=DEFAULT
;
; DEFAULT-DAG: @{{.+}} = {{.+}} #[[CUDABC:[0-9]+]]
; DEFAULT-DAG: @{{.+}} = {{.+}} #[[CUDAFB:[0-9]+]]
; DEFAULT-DAG: attributes #[[CUDABC]] = { kit_bc kit_tt(2) }
; DEFAULT-DAG: attributes #[[CUDAFB]] = { kit_fb kit_tt(2) }
;
; ------------------------------------------------------------------------------
; Check that the --tapir option overrides the default. The default is
; --tapir=cuda (integer value == 2)
;
; RUN: %kit-enc -tapir=hip %S/input/empty.ll \
; RUN:     | FileCheck %s --check-prefix=HIP
;
; HIP-DAG: @{{.+}} = {{.+}} #[[HIPBC:[0-9]+]]
; HIP-DAG: @{{.+}} = {{.+}} #[[HIPFB:[0-9]+]]
; HIP-DAG: attributes #[[HIPBC]] = { kit_bc kit_tt(4) }
; HIP-DAG: attributes #[[HIPFB]] = { kit_fb kit_tt(4) }
;
; ------------------------------------------------------------------------------
; Check that the --name option overrides the input module name. In this case,
; the input module name would be the name of this file, basic.ll
;
; RUN: %kit-enc --name="winnie-the-pooh" %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s --check-prefix=NAME
;
; NAME: ModuleID = 'winnie-the-pooh'
; NAME: !kit.module.device.module.flags = !{!{{[0-9]+}}, ![[NAMEMD:[0-9]+]]}
; NAME: [[NAMEMD]] = !{!"winnie-the-pooh"}
