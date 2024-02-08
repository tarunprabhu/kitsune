# The plugin should not be linked against LLVM libraries since these are
# expected to exist in the process loading the plugin. This doesn't work with
# DLLs on Windows. Kitsune will probably never support Windows, so this check is
# probably not necessary, but keep it around just in case.
if (WIN32 OR CYGWIN)
  message(FATAL_ERROR "Pass plugins are not supported on Windows")
endif ()

# Using add_llvm_library is only needed because we are building this within
# LLVM's build system.
add_llvm_pass_plugin(kit-pass-plugin-demo
  PassPluginDemo.cpp

  DEPENDS
  intrinsics_gen
  kitsune_generated

  BUILDTREE_ONLY
)

# Set the OUTPUT_NAME property because we need to retrieve it for use in the
# lit tests. We set the prefix to "" so "lib" is not prefixed to the shared
# library file name.
set_target_properties(kit-pass-plugin-demo PROPERTIES
  PREFIX ""
  OUTPUT_NAME "kit-pass-plugin-demo"
  LIBRARY_OUTPUT_DIRECTORY ${KITSUNE_EXAMPLES_BINARY_DIR})
