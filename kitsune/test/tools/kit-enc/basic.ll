; Check basic usage and command line options
;
; ------------------------------------------------------------------------------
; The default tapir target to add to the encoded module is cuda.
;
; RUN: %kit-enc %s | FileCheck %s --check-prefix=CUDA
;
; ------------------------------------------------------------------------------
; Check that the --tapir option works as expected when the value is the default,
; i.e. cuda.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | FileCheck %s --check-prefix=CUDA
;
; CUDA-DAG: @{{.+}} = {{.+}}, !kit.gv ![[MDBC:[0-9]+]]
; CUDA-DAG: @{{.+}} = {{.+}}, !kit.gv ![[MDDC:[0-9]+]]
;
; CUDA-DAG: ![[MDBC]] = distinct !{![[MDBC]], ![[BC:[0-9]+]]}
; CUDA-DAG: ![[BC]] = !{!"kit.gv.bit.code", i32 2}
; CUDA-DAG: ![[MDDC]] = distinct !{![[MDDC]], ![[DC:[0-9]+]]}
; CUDA-DAG: ![[DC]] = !{!"kit.gv.device.code", i32 2}
;
; ------------------------------------------------------------------------------
; Check that the --tapir option overrides the default. The default is
; --tapir=cuda (integer value == 2)
;
; RUN: %kit-enc -tapir=hip %s \
; RUN:     | FileCheck %s --check-prefix=HIP
;
; HIP-DAG: @{{.+}} = {{.+}}, !kit.gv ![[MDBC:[0-9]+]]
; HIP-DAG: @{{.+}} = {{.+}}, !kit.gv ![[MDDC:[0-9]+]]
;
; HIP-DAG: ![[MDBC]] = distinct !{![[MDBC]], ![[BC:[0-9]+]]}
; HIP-DAG: ![[BC]] = !{!"kit.gv.bit.code", i32 4}
; HIP-DAG: ![[MDDC]] = distinct !{![[MDDC]], ![[DC:[0-9]+]]}
; HIP-DAG: ![[DC]] = !{!"kit.gv.device.code", i32 4}
;
; ------------------------------------------------------------------------------
; The --tapir option can be provided more than once. In that case, one
; bit/device-code pair of globals must be created for each tapir target.
;
; RUN: %kit-enc --tapir=cuda --tapir=hip %s \
; RUN:     | FileCheck %s --check-prefix=MULTIPLE
;
; MULTIPLE-DAG: @{{.+}} = {{.+}}, !kit.gv ![[ATTRSBC2:[0-9]+]]
; MULTIPLE-DAG: @{{.+}} = {{.+}}, !kit.gv ![[ATTRSDC2:[0-9]+]]
; MULTIPLE-DAG: @{{.+}} = {{.+}}, !kit.gv ![[ATTRSBC4:[0-9]+]]
; MULTIPLE-DAG: @{{.+}} = {{.+}}, !kit.gv ![[ATTRSDC4:[0-9]+]]
;
; MULTIPLE-DAG: ![[ATTRSBC2]] = distinct !{![[ATTRSBC2]], ![[BC2:[0-9]+]]}
; MULTIPLE-DAG: ![[BC2]] = !{!"kit.gv.bit.code", i32 2}
; MULTIPLE-DAG: ![[ATTRSDC2]] = distinct !{![[ATTRSDC2]], ![[DC2:[0-9]+]]}
; MULTIPLE-DAG: ![[DC2]] = !{!"kit.gv.device.code", i32 2}
; MULTIPLE-DAG: ![[ATTRSBC4]] = distinct !{![[ATTRSBC4]], ![[BC4:[0-9]+]]}
; MULTIPLE-DAG: ![[BC4]] = !{!"kit.gv.bit.code", i32 4}
; MULTIPLE-DAG: ![[ATTRSDC4]] = distinct !{![[ATTRSDC4]], ![[DC4:[0-9]+]]}
; MULTIPLE-DAG: ![[DC4]] = !{!"kit.gv.device.code", i32 4}
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
; NAME: !kit.module = !{![[MD:[0-9]+]]}
; NAME: ![[MD]] = distinct !{![[MD]], ![[FLAGS:[0-9]+]]}
; NAME: ![[FLAGS]] = !{!"kit.module.device.module.flags", i32 {{[0-9]+}}, !"winnie-the-pooh"}
