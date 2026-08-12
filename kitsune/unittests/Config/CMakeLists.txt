set(LLVM_LINK_COMPONENTS
  KitCore
  Support
)

add_kitsune_unittest(KitConfig
  ConfigTest.cpp
)
