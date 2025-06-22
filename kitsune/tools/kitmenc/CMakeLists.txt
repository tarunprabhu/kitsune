set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Core
  IRReader
  KitCore
  KitSupport
  Support
)

add_llvm_tool(kitmenc
  kitmenc.cpp)
