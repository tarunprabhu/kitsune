set(LLVM_LINK_COMPONENTS
  Core
  Transforms
  Support)

add_llvm_tool(kitmenc
  kitmenc.cpp)
