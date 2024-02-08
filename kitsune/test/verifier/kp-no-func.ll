; If both a global variable containing embedded bitcode and a global variable
; containing kernel properties exist with the same tapir target, an appropriate
; function must exist in the embedded module.
;
; The command below does the following:
;
;   - Generate a file containing global variables for the embedded bitcode and
;     fat binary. The embedded bitcode is generated from this file
;
;   - Add a global variable that contains properties for the kernel. Note that
;     the value of the kit_kernel_props attribute is "g" whereas the embedded
;     bitcode contains a function named "f"
;
; ------------------------------------------------------------------------------
; RUN: ENC_2=`%kit-enc --tapir=cuda %s`
; RUN: MAX_2=`echo "${ENC_2}" \
; RUN:     | tail -n 1 \
; RUN:     | grep -oE "^[\!][0-9]+" \
; RUN:     | tail -c +2`
; RUN: KPID_2=`echo "${MAX_2} + 1" \
; RUN:     | bc`
; RUN: KP_2=`echo "${KPID_2} + 1" \
; RUN:     | bc`
;
; RUN: echo "${ENC_2}" \
; RUN:     | sed "\$a\@0 = constant i32 0, \!kit.gv \!${KPID_2}" \
; RUN:     | sed "\$a\\!${KPID_2} = distinct \!\{\!${KPID_2}, \!${KP_2}\}" \
; RUN:     | sed "\$a\\!${KP_2} = \!{\!\"kit.gv.kernel.properties\", i32 2, \!\"g\"\}" \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; ------------------------------------------------------------------------------
; RUN: ENC_4=`%kit-enc --tapir=hip %s`
; RUN: MAX_4=`echo "${ENC_4}" \
; RUN:     | tail -n 1 \
; RUN:     | grep -oE "^[\!][0-9]+" \
; RUN:     | tail -c +2`
; RUN: KPID_4=`echo "${MAX_4} + 1" \
; RUN:     | bc`
; RUN: KP_4=`echo "${KPID_4} + 1" \
; RUN:     | bc`
;
; RUN: echo "${ENC_4}" \
; RUN:     | sed "\$a\@0 = constant i32 0, \!kit.gv \!${KPID_4}" \
; RUN:     | sed "\$a\\!${KPID_4} = distinct \!\{\!${KPID_4}, \!${KP_4}\}" \
; RUN:     | sed "\$a\\!${KP_4} = \!{\!\"kit.gv.kernel.properties\", i32 4, \!\"g\"\}" \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; ------------------------------------------------------------------------------
;
; CHECK: attribute 'kit.gv.kernel.properties': invalid value at index '1'
; CHECK-SAME: Kernel function does not exist in embedded module

define void @f() {
  ret void
}
