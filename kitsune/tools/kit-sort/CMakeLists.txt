set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  Core
  IRReader
  KitCore
  KitSupport
  Support
)

add_llvm_tool(kit-sort
  kit-sort.cpp

  DEPENDS
  intrinsics_gen
  kitsune_generated)
