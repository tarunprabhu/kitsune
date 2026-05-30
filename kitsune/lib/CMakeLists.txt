# This subdirectories here are a mix of those destined to be part of libLLVM.so,
# and those that will be compiled into libraries to be used by the frontends,
# clang, flang and anything else that we may care to support at some point.
#
# This directory is only entered from llvm/lib/CMakeLists.txt, and expects to
# compile support libraries that will become part of libLLVM.so. Therefore,
# we only descend into the non-frontend-related subdirectories here.
#
# Clang/, and any other frontend-related subdirectories, will be entered
# directly from, say, clang/lib/CMakeLists.txt.

add_subdirectory(Analysis)
add_subdirectory(CodeGen)
add_subdirectory(Core)
add_subdirectory(Passes)
add_subdirectory(Support)
add_subdirectory(Targets)
add_subdirectory(Transforms)
