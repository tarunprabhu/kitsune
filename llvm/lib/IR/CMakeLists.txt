add_llvm_component_library(LLVMCore
  AbstractCallSite.cpp
  AsmWriter.cpp
  Assumptions.cpp
  Attributes.cpp
  AutoUpgrade.cpp
  BasicBlock.cpp
  BuiltinGCs.cpp
  Comdat.cpp
  ConstantFold.cpp
  ConstantFPRange.cpp
  ConstantRange.cpp
  ConstantRangeList.cpp
  Constants.cpp
  ConvergenceVerifier.cpp
  Core.cpp
  CycleInfo.cpp
  DIBuilder.cpp
  DataLayout.cpp
  DebugInfo.cpp
  DebugInfoMetadata.cpp
  DIExpressionOptimizer.cpp
  DebugProgramInstruction.cpp
  DebugLoc.cpp
  DiagnosticHandler.cpp
  DiagnosticInfo.cpp
  DiagnosticPrinter.cpp
  Dominators.cpp
  EHPersonalities.cpp
  FPEnv.cpp
  Function.cpp
  GCStrategy.cpp
  GVMaterializer.cpp
  Globals.cpp
  Intrinsics.cpp
  IRBuilder.cpp
  IRPrintingPasses.cpp
  SSAContext.cpp
  InlineAsm.cpp
  Instruction.cpp
  Instructions.cpp
  IntrinsicInst.cpp
  LLVMContext.cpp
  LLVMContextImpl.cpp
  LLVMRemarkStreamer.cpp
  LegacyPassManager.cpp
  MDBuilder.cpp
  MemoryModelRelaxationAnnotations.cpp
  Mangler.cpp
  Metadata.cpp
  Module.cpp
  ModuleSummaryIndex.cpp
  Operator.cpp
  OptBisect.cpp
  Pass.cpp
  PassInstrumentation.cpp
  PassManager.cpp
  PassRegistry.cpp
  PassTimingInfo.cpp
  PrintPasses.cpp
  ProfDataUtils.cpp
  SafepointIRVerifier.cpp
  ProfileSummary.cpp
  PseudoProbe.cpp
  ReplaceConstant.cpp
  Statepoint.cpp
  StructuralHash.cpp
  Type.cpp
  TypedPointerType.cpp
  TypeFinder.cpp
  Use.cpp
  User.cpp
  Value.cpp
  ValueSymbolTable.cpp
  VectorBuilder.cpp
  VectorTypeUtils.cpp
  Verifier.cpp
  VFABIDemangler.cpp
  RuntimeLibcalls.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/IR

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}

  DEPENDS
  intrinsics_gen

  # Including BitReader here results in a circular dependence since BitReader
  # requires Core. This is a known issue in Kitsune. We need BitReader in order
  # to verify embedded modules. The fix would be to move the verifier out of
  # Core, which will make it much harder to keep up with LLVM, so we resign
  # ourselves to having a cycle here.
  #
  # We want all IR verification to be carried out in the main verifier which is
  # in this library. Doing so cleanly requires utilities from KitCore which
  # already depends on this. We now have yet another circular dependence.
  # Ideally, we would like the verifier to live outside LLVMCore in which case,
  # we could easily break this dependence cycle, but that would cause us no end
  # of problems when merging, so, regrettably, we introduce (yet another)
  # circular dependence.
  #
  LINK_COMPONENTS
  BinaryFormat
  BitReader
  Demangle
  KitCore
  KitSupport
  Remarks
  Support
  TargetParser
  )
