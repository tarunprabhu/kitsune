include(KitsuneTableGen)

kitsune_tablegen(LoopAttrs.td LoopAttrs.inc
  -gen-kitsune-loop-attrs
  TARGET kitsune_generated_loop_attrs)

add_dependencies(kitsune_generated
  kitsune_generated_loop_attrs)
