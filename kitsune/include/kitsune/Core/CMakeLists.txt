include(KitsuneTableGen)

kitsune_tablegen(LoopAttrs.td LoopAttrs.inc
  -gen-kitsune-loop-attrs
  TARGET kitsune_generated_loop_attrs)

kitsune_tablegen(ModuleAttrs.td ModuleAttrs.inc
  -gen-kitsune-module-attrs
  TARGET kitsune_generated_module_attrs)

add_dependencies(kitsune_generated
  kitsune_generated_loop_attrs
  kitsune_generated_module_attrs)
