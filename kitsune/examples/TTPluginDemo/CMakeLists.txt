# The plugin should not be linked against LLVM libraries since these are
# expected to exist in the process loading the plugin. This doesn't work with
# DLLs on Windows. Kitsune will probably never support Windows, so this check is
# probably not necessary, but keep it around just in case.
if (WIN32 OR CYGWIN)
  message(FATAL_ERROR "Tapir target plugins are not supported on Windows")
endif ()

# Using add_llvm_library is only needed because we are building this within
# LLVM's build system. The README.md associate with this example will provide
# more information on how this can be built out-of-tree.
add_llvm_pass_plugin(kit-ttplugin-demo
  TTPluginDemo.cpp
  DEPENDS
  intrinsics_gen
  BUILDTREE_ONLY
)

# Set the OUTPUT_NAME property because we need to retrieve it for use in the
# lit tests
set_target_properties(kit-ttplugin-demo PROPERTIES
  PREFIX ""
  OUTPUT_NAME "kit-ttplugin-demo"
  LIBRARY_OUTPUT_DIRECTORY ${KITSUNE_EXAMPLES_BINARY_DIR})
