# This configuration is for the lit tests for Kitsune's runtime.

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  "LLVM_EXTERNAL_LIT"
  "LLVM_LIT_TOOLS_DIR"
  "LLVM_TOOLS_DIR"
  "KITSUNE_BINARY_DIR"
  "KITSUNE_SOURCE_DIR"
)
