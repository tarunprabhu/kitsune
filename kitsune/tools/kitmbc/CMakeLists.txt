set(LLVM_LINK_COMPONENTS
  Core
  Transforms
  Support)

add_llvm_tool(kitmbc
  kitmbc.cpp)
