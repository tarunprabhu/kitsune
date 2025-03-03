set(LLVM_LINK_COMPONENTS nativecodegen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_unittest(MLIRTargetLLVMTests
  SerializeNVVMTarget.cpp
  SerializeROCDLTarget.cpp
  SerializeToLLVMBitcode.cpp
DEPENDS
  ${dialect_libs}
)

llvm_map_components_to_libnames(llvm_libs nativecodegen)
# This might be a bug in MLIR. If LLVM_LINK_LLVM_DYLIB is ON, we get a duplicate
# definition error on MLIRTargetLLVMTests because some the command line options
# are registered twice - one from the static libraries in ${llvm_libs} and again
# from libLLVM.so. Only use ${llvm_libs} when LLVM_LINK_LLVM_DYLIB is not ON.
if (LLVM_LINK_LLVM_DYLIB)
  set(llvm_libs "")
endif()

target_link_libraries(MLIRTargetLLVMTests
  PRIVATE
  MLIRTargetLLVM
  MLIRNVVMTarget
  MLIRROCDLTarget
  MLIRGPUDialect
  MLIRNVVMDialect
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
)

if (DEFINED LLVM_NATIVE_TARGET)
  target_compile_definitions(MLIRTargetLLVMTests
    PRIVATE
    -DLLVM_NATIVE_TARGET_TEST_ENABLED=1
  )
else()
  target_compile_definitions(MLIRTargetLLVMTests
    PRIVATE
    -DLLVM_NATIVE_TARGET_TEST_ENABLED=0
  )
endif()
