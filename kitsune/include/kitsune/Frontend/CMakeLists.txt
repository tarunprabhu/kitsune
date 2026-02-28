include(KitsuneTableGen)

kitsune_tablegen(Diagnostics.td Diagnostics.inc
  -gen-kitsune-diags
  TARGET kitsune_generated_diags)

add_dependencies(kitsune_generated
  kitsune_generated_diags)
