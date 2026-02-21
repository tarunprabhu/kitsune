# Kitsune is registered as a project and will show up as a "tool" alongside
# clang. But we actually want this configuration file in a place where clang
# will pick it up.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake
  ${CMAKE_BINARY_DIR}/include/kitsune/Config/config.h
  @ONLY)
