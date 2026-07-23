include(KitsuneTableGen)

function(gen_inc source dest cmd)
  set(target "kitsune_generated_${dest}")
  kitsune_tablegen(${source} ${dest} ${cmd} TARGET ${target})

  add_dependencies(kitsune_generated ${target})
endfunction()

gen_inc(ArgAttrs.td ArgAttrs.inc -gen-kit-arg-attr-header)
gen_inc(Diagnostics.td Diagnostics.inc -gen-kit-diag-header)
gen_inc(FuncAttrs.td FuncAttrs.inc -gen-kit-func-attr-header)
gen_inc(GVAttrs.td GVAttrs.inc -gen-kit-gv-attr-header)
gen_inc(InstAttrs.td InstAttrs.inc -gen-kit-inst-attr-header)
gen_inc(${LLVM_MAIN_INCLUDE_DIR}/llvm/IR/Intrinsics.td IntrLibFuncMap.inc -gen-kit-intr-libfunc-map)
gen_inc(LibFuncs.td LibFuncs.inc -gen-kit-libfunc-header)
gen_inc(LoopAttrs.td LoopAttrs.inc -gen-kit-loop-attr-header)
gen_inc(ModuleAttrs.td ModuleAttrs.inc -gen-kit-module-attr-header)
