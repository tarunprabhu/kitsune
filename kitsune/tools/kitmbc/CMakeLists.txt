set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  BitWriter
  CodeGen
  Core
  IRReader
  KitCore
  KitSupport
  Support
)

add_llvm_tool(kitmbc
  kitmbc.cpp)
