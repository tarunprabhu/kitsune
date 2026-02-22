include(KitsuneTableGen)

kitsune_tablegen(LoopAttrs.td LoopAttrs.inc
  -gen-kitsune-loop-attrs
  TARGET kitsune_generated_loop_attrs)

# The sole purpose of this target is to depend on the targets for entities
# created by kitsune_tablegen. Other targets outside Kitsune, particularly in
# LLVM, can then depend on just a single target from Kitsune.
add_custom_target(kitsune_generated_attrs)
add_dependencies(kitsune_generated_attrs
  kitsune_generated_loop_attrs)
