# The plugin should not be linked against LLVM libraries since these are
# expected to exist in the process loading the plugin. This doesn't work with
# DLLs on Windows. Kitsune will probably never support Windows, so this check is
# probably not necessary, but keep it around just in case.
if (WIN32 OR CYGWIN)
  message(FATAL_ERROR "Pass plugins are not supported on Windows")
endif ()

add_llvm_pass_plugin(kit-emb-pass-plugin-demo
  EmbModulePassPluginDemo.cpp
  DEPENDS
  intrinsics_gen
  BUILDTREE_ONLY)

# Set the OUTPUT_NAME property because we need to retrieve it for use in the
# lit tests. We set the prefix to "" so "lib" is not prefixed to the shared
# library file name.
set_target_properties(kit-emb-pass-plugin-demo PROPERTIES
  PREFIX ""
  OUTPUT_NAME "kit-emb-pass-plugin-demo"
  LIBRARY_OUTPUT_DIRECTORY ${KITSUNE_EXAMPLES_BINARY_DIR})
