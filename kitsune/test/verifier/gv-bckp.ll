; A global variable cannot have both the bit.code and kernel.properties
; attributes.
;
; ------------------------------------------------------------------------------
;
; This generates valid embedded bitcode using kit-enc since global variables
; containing bitcode must have valid initializers. The output of the kit-enc
; invocation will be as follows:
;
;     @bc = unnamed_addr constant [<N> x i8] c"<CODE>", !kit.gv.bit.code !0
;     @fb = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0
;
;     !0 = !{i32 2}
;
; The first sed command looks for unnamed_addr since it will only match the
; line containing embedded bitcode. It then appends the kernel.properties
; attribute to the line. The output will be as follows:
;
;     @bc = unnamed_addr constant [<N> x i8] c"<CODE>", !kit.gv.bit.code !0, !kit.gv.kernel.properties !1
;     @fb = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0
;
;     !0 = !{i32 2}
;
; The second sed command appends the definition of !1. The final result that
; is passed to llvm-as will be as follows:
;
;     @bc = unnamed_addr constant [<N> x i8] c"<CODE>", !kit.gv.bit.code !0, !kit.gv.kernel.properties !1
;     @fb = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0
;
;     !0 = !{i32 2}
;     !1 = !{!"<STRING>"}
;
; ------------------------------------------------------------------------------
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | sed -E 's/unnamed_addr(.+)/\1, \!kit.gv.kernel.properties \!1/g' \
; RUN:     | sed '$a\\!1 = \!{\!"selwyn"\}' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | sed -E 's/unnamed_addr(.+)/\1, \!kit.gv.kernel.properties \!1/g' \
; RUN:     | sed '$a\\!1 = \!{\!"imperial"\}' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: Attributes 'bit.code' and 'kernel.properties' are incompatible

define void @kf() {
  ret void
}
