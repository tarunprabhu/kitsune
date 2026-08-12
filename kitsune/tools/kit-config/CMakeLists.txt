set(LLVM_LINK_COMPONENTS
  KitCore
  Support
)

add_llvm_tool(kit-config
  kit-config.cpp)
