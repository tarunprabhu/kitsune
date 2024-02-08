# Setup a Kitsune frontend symlink (kitcc, kit++ etc.).
#
#   NAME   The name of the frontend. This is also the name of the symlink that
#          will be created
#
#   DEST   The name of the target of the symlink. It *MUST NOT* be the full path
#          to the destination since it is assumed to be in
#          ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}, which is where the symlink will
#          be created
#
#   DEP    The name of the cmake target that created DEST
#
function(setup_frontend_symlink name dest dep)
  set(symlink "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}")
  add_custom_target(${name} ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${dest} ${symlink}
    COMMENT "Creating symlink ${name} to ${dest}"
    VERBATIM USES_TERMINAL)
  add_dependencies(${name} ${dep})
  install(FILES
    ${symlink}
    DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()
