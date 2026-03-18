include(KitsuneTableGen)

kitsune_tablegen(InstAttrs.td InstAttrs.inc
  -gen-kitsune-inst-attrs
  TARGET kitsune_generated_inst_attrs)

kitsune_tablegen(LoopAttrs.td LoopAttrs.inc
  -gen-kitsune-loop-attrs
  TARGET kitsune_generated_loop_attrs)

kitsune_tablegen(ModuleAttrs.td ModuleAttrs.inc
  -gen-kitsune-module-attrs
  TARGET kitsune_generated_module_attrs)

add_dependencies(kitsune_generated
  kitsune_generated_inst_attrs
  kitsune_generated_loop_attrs
  kitsune_generated_module_attrs)
