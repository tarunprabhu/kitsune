set(LLVM_LINK_COMPONENTS
  KitCore
  KitSupport
  Support
)

add_kitsune_unittest(KitSupport
  FromIntTest.cpp
  FromStringTest.cpp
  MaybeBoolTest.cpp
  TypeTraitsTest.cpp
)
