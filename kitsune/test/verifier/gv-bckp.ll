; A global variable cannot have both the bit.code and kernel.properties
; attributes.
;
; ------------------------------------------------------------------------------
;
; This generates valid embedded bitcode using kit-enc since global variables
; containing bitcode must have valid initializers. The output of the kit-enc
; invocation will be as follows:
;
;     @bc = unnamed_addr constant [<N> x i8] c"<CODE>", !kit.gv !0
;     @fb = constant [0 x i8] zeroinitializer, !kit.gv !1
;
;     !0 = distinct !{!0, !2}
;     !1 = distinct !{!1, !3}
;     !2 = !{!"kit.gv.bit.code", i32 2}
;     !3 = !{!"kit.gv.device.code", i32 2}
;
; MAX_* is the one greater than the largest metadata node label in the IR. The
; largest metadata node is expected to be on the last line of the IR. For the
; code above, this will be '4'.
;
; MD_* is the metadata node containing the list of Kitsune-specific attributes
; for the global variable containing bitcode. We find it by looking for the
; `unnamed_addr` property since it will only match the line containing embedded
; bitcode. Here, this will be '0'.
;
; The first sed commands adds an entry into the attribute list for the kernel
; properties attribute that will be added later.
;
;     !0 = !{!0, !4, !1}
;
; The second sed command appends the actual kernel properties attribute. The
; TTID should match that on the bitcode global.
;
;     !4 = !{!"kit.gv.kernel.properties", i32 <TTID>, !"<KERNEL_NAME>"}
;
; ------------------------------------------------------------------------------
;
; RUN: ENC_2=`%kit-enc --tapir=cuda %s`
; RUN: MAX_2=`echo "${ENC_2}" \
; RUN:     | tail -n 1 \
; RUN:     | grep -oE "^[\!][0-9]+" \
; RUN:     | tail -c +2 \
; RUN:     | xargs -I{} echo "{}+1" \
; RUN:     | bc`
; RUN: MD_2=`echo "${ENC_2}" \
; RUN:     | grep -E "unnamed_addr constant .+ c[\"]" \
; RUN:     | grep -oE "[0-9]+$"`
;
; RUN: echo "${ENC_2}" \
; RUN:     | sed "s/\!${MD_2},/\!${MD_2}, \!${MAX_2},/g" \
; RUN:     | sed "\$a\\!${MAX_2} = \!\{\!\"kit.gv.kernel.properties\", i32 2, \!\"selwyn\"\}" \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; ------------------------------------------------------------------------------
;
; RUN: ENC_4=`%kit-enc --tapir=hip %s`
; RUN: MAX_4=`echo "${ENC_4}" \
; RUN:     | tail -n 1 \
; RUN:     | grep -oE "^[\!][0-9]+" \
; RUN:     | tail -c +2 \
; RUN:     | xargs -I{} echo "{}+1" \
; RUN:     | bc`
; RUN: MD_4=`echo "${ENC_4}" \
; RUN:     | grep -E "unnamed_addr constant .+ c[\"]" \
; RUN:     | grep -oE "[0-9]+$"`
;
; RUN: echo "${ENC_4}" \
; RUN:     | sed "s/\!${MD_4},/\!${MD_4}, \!${MAX_4},/g" \
; RUN:     | sed "\$a\\!${MAX_4} = \!\{\!\"kit.gv.kernel.properties\", i32 4, \!\"imperial\"\}" \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; ------------------------------------------------------------------------------
;
; CHECK-DAG: attribute 'kit.gv.bit.code': not compatible with 'kit.gv.kernel.properties'
; CHECK-DAG: attribute 'kit.gv.kernel.properties': not compatible with 'kit.gv.bit.code'

define void @kf() {
  ret void
}
