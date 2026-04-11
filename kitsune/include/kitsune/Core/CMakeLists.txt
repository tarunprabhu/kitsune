include(KitsuneTableGen)

kitsune_tablegen(ArgAttrs.td ArgAttrs.inc
  -gen-kit-arg-attr-header
  TARGET kitsune_generated_arg_attrs)

kitsune_tablegen(FuncAttrs.td FuncAttrs.inc
  -gen-kit-func-attr-header
  TARGET kitsune_generated_func_attrs)

kitsune_tablegen(GVAttrs.td GVAttrs.inc
  -gen-kit-gv-attr-header
  TARGET kitsune_generated_gv_attrs)

kitsune_tablegen(InstAttrs.td InstAttrs.inc
  -gen-kit-inst-attr-header
  TARGET kitsune_generated_inst_attrs)

kitsune_tablegen(LoopAttrs.td LoopAttrs.inc
  -gen-kit-loop-attr-header
  TARGET kitsune_generated_loop_attrs)

kitsune_tablegen(ModuleAttrs.td ModuleAttrs.inc
  -gen-kit-module-attr-header
  TARGET kitsune_generated_module_attrs)

add_dependencies(kitsune_generated
  kitsune_generated_arg_attrs
  kitsune_generated_func_attrs
  kitsune_generated_gv_attrs
  kitsune_generated_inst_attrs
  kitsune_generated_loop_attrs
  kitsune_generated_module_attrs)
