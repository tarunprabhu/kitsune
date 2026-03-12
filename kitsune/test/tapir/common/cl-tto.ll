; Check that the tapir target options are set correctly depending on the options
; passed to LLVM's opt utility. These options are common to all tapir targets
; Since the serial tapir target is guaranteed to be built, we use that here.
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     -passes="tapir-lowering<O2>" -dump-tapir-target-options \
; RUN:     | FileCheck %s -check-prefixes ALL,DEFAULT
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     --tapir-verbose \
; RUN:     -passes="tapir-lowering<O2>" -dump-tapir-target-options \
; RUN:     | FileCheck %s -check-prefixes ALL,COMPILER
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     --kitrt-verbose \
; RUN:     -passes="tapir-lowering<O2>" -dump-tapir-target-options \
; RUN:     | FileCheck %s -check-prefixes ALL,RUNTIME
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     -passes="tapir-lowering<O1>" -dump-tapir-target-options \
; RUN:     | FileCheck %s --check-prefixes ALL,O1
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     -passes="tapir-lowering<O3>" -dump-tapir-target-options \
; RUN:     | FileCheck %s --check-prefixes ALL,O3
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     -fp-contract=off \
; RUN:     -passes="tapir-lowering<O2>" -dump-tapir-target-options \
; RUN:     | FileCheck %s -check-prefixes ALL,FP_STRICT
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     -fp-contract=on \
; RUN:     -passes="tapir-lowering<O2>" -dump-tapir-target-options \
; RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     -fp-contract=fast \
; RUN:     -passes="tapir-lowering<O2>" -dump-tapir-target-options \
; RUN:     | FileCheck %s -check-prefixes ALL,FP_FAST
;
; ALL:          Tapir target options
; DEFAULT:      Compiler verbose: 0
; COMPILER:     Compiler verbose: 1
; COMPILER:     Runtime verbose: 1
; RUNTIME:      Compiler verbose: 0
; RUNTIME:      Runtime verbose: 1
; O1:           Optimization level: O1
; O3:           Optimization level: O3
; FP_STRICT:    FP fusion: strict
; FP_STANDARD:  FP fusion: standard
; FP_FAST:      FP fusion: fast
