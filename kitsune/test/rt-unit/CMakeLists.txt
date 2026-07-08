# This configuration is for the unit tests for Kitsune's runtime.

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  "KITRT_BINARY_DIR"
  "KITRT_SOURCE_DIR"
  "KITSUNE_BINARY_DIR"
  "KITSUNE_SOURCE_DIR"
)
