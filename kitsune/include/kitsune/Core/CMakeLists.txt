include(KitsuneTableGen)

function(add_kitsune_tablegen source dest cmd target)
  kitsune_tablegen(${source} ${dest} ${cmd} TARGET ${target})

  add_dependencies(kitsune_generated ${target})
endfunction()

add_kitsune_tablegen(ArgAttrs.td ArgAttrs.inc
  -gen-kit-arg-attr-header
  kitsune_generated_arg_attrs)

add_kitsune_tablegen(Diagnostics.td Diagnostics.inc
  -gen-kit-diag-header
  kitsune_generated_diags)

add_kitsune_tablegen(FuncAttrs.td FuncAttrs.inc
  -gen-kit-func-attr-header
  kitsune_generated_func_attrs)

add_kitsune_tablegen(GVAttrs.td GVAttrs.inc
  -gen-kit-gv-attr-header
  kitsune_generated_gv_attrs)

add_kitsune_tablegen(InstAttrs.td InstAttrs.inc
  -gen-kit-inst-attr-header
  kitsune_generated_inst_attrs)

add_kitsune_tablegen(LibFuncs.td LibFuncs.inc
  -gen-kit-libfunc-header
  kitsune_generated_lib_funcs)

add_kitsune_tablegen(LoopAttrs.td LoopAttrs.inc
  -gen-kit-loop-attr-header
  kitsune_generated_loop_attrs)

add_kitsune_tablegen(ModuleAttrs.td ModuleAttrs.inc
  -gen-kit-module-attr-header
  kitsune_generated_module_attrs)
