set(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kit-objenc.py)
set(DEST ${CMAKE_BINARY_DIR}/bin/kit-objenc)

add_custom_command(
  OUTPUT ${DEST}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SOURCE} ${DEST}
  DEPENDS ${SOURCE}
  VERBATIM USES_TERMINAL)

add_custom_target(kit-objenc ALL
  DEPENDS ${DEST} llvm-strip)
