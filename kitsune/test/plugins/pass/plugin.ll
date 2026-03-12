; Check that the passes in a pass plugin are registered at the correct places
; in the pass pipeline.
;
; RUN: opt --tapir=serial -O1 -o /dev/null -debug-pass-manager %s \
; RUN:     --load-pass-plugin=%kit-pass-plugin-demo 2>&1 \
; RUN:     | FileCheck %s --check-prefix=PASSES
;
; PASSES: PreTapirEarlyPass
; PASSES: PreTapirLatePass
; PASSES: LoopSpawningPass
; PASSES: PostTapirEarlyPass
; PASSES: PostTapirLatePass
; PASSES: GenerateCtorsPass
; PASSES: PostTapirLastPass
;
; ------------------------------------------------------------------------------
; If only the pre-tapir-early pass is requested, the other passes in the plugin
; should not be run.
;
; RUN: opt --tapir=serial -disable-output -debug-pass-manager %s \
; RUN:     --load-pass-plugin=%kit-pass-plugin-demo \
; RUN:     -passes='pre-tapir-early' 2>&1 \
; RUN:     | FileCheck %s --check-prefix=PRE-EARLY
;
; PRE-EARLY: PreTapirEarlyPass on [module]
; PRE-EARLY-NEXT: VerifierPass on [module]
; PRE-EARLY-NEXT: VerifierAnalysis on [module]
; PRE-EARLY-NOT: {{^.+$}}
;
; ------------------------------------------------------------------------------
; If only the pre-tapir-late pass is requested, the other passes in the plugin
; should not be run.
;
; RUN: opt --tapir=serial -disable-output -debug-pass-manager %s \
; RUN:     --load-pass-plugin=%kit-pass-plugin-demo \
; RUN:     -passes='pre-tapir-late' 2>&1 \
; RUN:     | FileCheck %s --check-prefix=PRE-LATE
;
; PRE-LATE: PreTapirLatePass on [module]
; PRE-LATE-NEXT: VerifierPass on [module]
; PRE-LATE-NEXT: VerifierAnalysis on [module]
; PRE-LATE-NOT: {{^.+$}}
;
; ------------------------------------------------------------------------------
; If only the post-tapir-early pass is requested, the other passes in the plugin
; should not be run.
;
; RUN: opt --tapir=serial -disable-output -debug-pass-manager %s \
; RUN:     --load-pass-plugin=%kit-pass-plugin-demo \
; RUN:     -passes='post-tapir-early' 2>&1 \
; RUN:     | FileCheck %s --check-prefix=POST-EARLY
;
; POST-EARLY: PostTapirEarlyPass on [module]
; POST-EARLY-NEXT: VerifierPass on [module]
; POST-EARLY-NEXT: VerifierAnalysis on [module]
; POST-EARLY-NOT: {{^.+$}}
;
; ------------------------------------------------------------------------------
; If only the post-tapir-late pass is requested, the other passes in the plugin
; should not be run.
;
; RUN: opt --tapir=serial -disable-output -debug-pass-manager %s \
; RUN:     --load-pass-plugin=%kit-pass-plugin-demo \
; RUN:     -passes='post-tapir-late' 2>&1 \
; RUN:     | FileCheck %s --check-prefix=POST-LATE
;
; POST-LATE: PostTapirLatePass on [module]
; POST-LATE-NEXT: VerifierPass on [module]
; POST-LATE-NEXT: VerifierAnalysis on [module]
; POST-LATE-NOT: {{^.+$}}
;
; ------------------------------------------------------------------------------
; If only the post-tapir-last pass is requested, the other passes in the plugin
; should not be run.
;
; RUN: opt --tapir=serial -disable-output -debug-pass-manager %s \
; RUN:     --load-pass-plugin=%kit-pass-plugin-demo \
; RUN:     -passes='post-tapir-last' 2>&1 \
; RUN:     | FileCheck %s --check-prefix=POST-LAST
;
; POST-LAST: PostTapirLastPass on [module]
; POST-LAST-NEXT: VerifierPass on [module]
; POST-LAST-NEXT: VerifierAnalysis on [module]
; POST-LAST-NOT: {{^.+$}}

