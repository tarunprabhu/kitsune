//===- GPUTTLoop.h - GPU-centric loop outline processors -------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base classes for loop outline processors used by the 'cuda' and 'hip' tapir
// targets.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TARGETS_GPUTT_LOOP_H
#define KITSUNE_TARGETS_GPUTT_LOOP_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {

class BlockAddress;
class Constant;
class GlobalAlias;
class GlobalIFunc;
class GlobalVariable;
class GlobalValue;

/// Base class shared by the loop outline processors of GPU-centric tapir
/// targets.
class GPUTTLoopBase : public LoopOutlineProcessor {
protected:
  /// The ID of tapir target that created this loop outline processor.
  TTID tt;

  /// The "host" module that contains the tapir loop being outlined.
  Module &hostM;

  /// The module into which the tapir loops are outlined. This will eventually
  /// be compiled to GPU machine code.
  Module &devM;

  /// The name of the kernel into which the loop is outlined.
  std::string kernelName;

  /// The depth of the loop nest that is being outlined. This must be in the
  /// range [1,3].
  unsigned kernelDepth = 0;

  /// The GlobalValue's used in the loop that is being outlined. This includes
  /// functions, global variables, aliases and ifunc's.
  SmallSet<GlobalValue *, 8> usedGlobalValues;

private:
  /// Populate the \ref usedGlobalValues member with the global values used in
  /// the initializer of the global variable \p g. \p g will also be added to
  /// \ref usedGlobalValues.
  void populateUsedGlobalValues(GlobalVariable &g);

  /// Populate the \ref usedGlobalValues member with the global values used in
  /// the global ifunc \p g.
  void populateUsedGlobalValues(GlobalIFunc &g);

  /// Populate the \ref usedGlobalValues member with the global values used in
  /// the initializer of the global alias \p g.
  void populateUsedGlobalValues(GlobalAlias &g);

  /// Populate the \ref usedGlobalValues member with the global values used in
  /// the block address \p blkAddr.
  void populateUsedGlobalValues(BlockAddress &blkaddr);

  /// Populate the \ref usedGlobalValues member with the global values used in
  /// the constant \p c.
  void populateUsedGlobalValues(Constant &c);

  /// Generate calls, in the host module \ref hostM, to copy non-constant
  /// globals. Calls to the \p copyFn intrinsic function will be inserted.
  void copyNonConstGlobals(IRBuilder<> &builder, Intrinsic::ID copyFn);

  /// If the global variable, \p g, is constant, clone it into the device
  /// module, \ref devM, along with its initializer with internal linkage. If
  /// the global variable, \p g, is not constant, clone it into the device
  /// module, \ref devM. with internal linkage. The cloned global is
  /// zero-initialized. But this will be overwritten at run-time. In each case,
  /// the cloned global will be a constant if and only if \p g is a constant.
  /// Returns the newly created global variable.
  GlobalVariable *cloneGlobalVariable(GlobalVariable &g);

protected:
  /// Populate the \ref usedGlobalValues member with the global values used in
  /// the basic block \p bb.
  void populateUsedGlobalValues(BasicBlock &bb);

  /// Populate the \ref usedGlobalValues member with the global values used in
  /// the function \p f.
  void populateUsedGlobalValues(Function &f);

  /// Populate the \ref usedGlobalValues member with the global values used in
  /// the loop \p loop. This will include the global values included in any
  /// subloops of \ploop
  void populateUsedGlobalValues(Loop &loop);

  /// Generate calls, in the host module \ref hostM, to copy non-constant
  /// globals, which are used in the outlined tapir loop, from device to host
  /// after launching the kernel.
  void copyNonConstGlobalsDToH(IRBuilder<> &builder);

  /// Generate calls, in the host module \ref hostM, to copy non-constant
  /// globals, which are used in the outlined tapir loop, from host to device
  /// before launching the kernel.
  void copyNonConstGlobalsHToD(IRBuilder<> &builder);

  /// Clone global aliases that are used in outlined tapir loop from the host
  /// module, \ref hostM, to the device module, \ref devM
  /// \param vmap The ValueMap containing mappings for the global values that
  ///             are to be cloned. This will be updated with mappings for the
  ///             cloned global aliases
  void cloneUsedGlobalAliases(ValueToValueMapTy &vmap);

  /// Clone global variables that are used in outlined tapir loop from the host
  /// module, \ref hostM, to the device module, \ref devM.
  /// \param vmap The ValueMap containing mappings for the global values that
  ///             are to be cloned. This will be updated with mappings for the
  ///             cloned global variables
  void cloneUsedGlobalVariables(ValueToValueMapTy &vmap);

  /// Clone functions that are transitively reachable from outlined tapir loops
  /// from the host module, \ref hostM, to the device module, \ref devM.
  /// \param vmap The ValueMap containing mappings for the global values that
  ///             are to be cloned. This will be updated with mappings for the
  ///             cloned functions.
  void cloneReachableFuncs(ValueToValueMapTy &vmap);

  /// Clone IFuncs that are transitively reachable from outlined tapir loops
  /// from the host module, \ref hostM, to the device module, \ref devM
  /// \param vmap The ValueMap containing mappings for the global values that
  ///             are to be cloned. This will be updated with mappings for the
  ///             cloned ifuncs.
  void cloneReachableIFuncs(ValueToValueMapTy &vmap);

  /// Get the global variable in the device module that corresponds to the
  /// global variable \g in the host. Since the global variables in the device
  /// module may be in a different address space from that in the global, the
  /// values in the vmap may be wrapped in a constant expression that casts
  /// away the address space. These have to be stripped away to get the
  /// underlying global.
  GlobalVariable *getDevGlobal(GlobalVariable *g,
                               const ValueToValueMapTy &vmap);

  /// Given a compare instruction, \p inst, that is the condition of a tapir
  /// loop, return the operand of the instruction that matches the value \p v.
  /// At least one of the instruction operands is expected to match \p v.
  unsigned getOpIndex(const Instruction &inst, Value *v);

  /// Get the grainsize to use. The default implementation always returns 1.
  virtual Value *getGrainsize(Type *ty);

  /// Set the correct calling convention on the outlined kernel function, \p f
  /// into which a tapir loop has been outlined.
  virtual void setKernelFuncCallingConv(Function &f) {}

  /// Set the correct linkage on an outlined kernel function, \p f into which a
  /// tapir loop has been outlined. The default implementation will set the
  /// linkage to external to prevent the outlined function from being DCE'ed if
  /// the device module is optimized.
  virtual void setKernelFuncLinkage(Function &f);

  /// Set the correct visibility on an outlined kernel function, \p f into which
  /// a kernel function has been outlined.
  virtual void setKernelFuncVisibility(Function &f) {}

  /// Set the correct attributes on an outlined kernel function, \p f into which
  /// a tapir loop has been outlined. This may remove existing attributes from
  /// the function, change the values of others, and add new ones as
  /// appropriate. The default implementation does nothing i.e. all function
  /// attributes will remain unchanged.
  virtual void setKernelFuncAttrs(Function &f) {}

  /// A kernel function may require module-level attributes to be set. This will
  /// set such attributes for the kernel function, \p f into which a tapir loop
  /// has been outlined.
  virtual void setModuleAttrsForKernelFunc(Function &f) {}

  /// Convert the induction variables in the outlined loop to use the GPU
  /// kernel's thread and block indices. The default implementation performs the
  /// standard calculation assuming that the launch will be on the X, Y and
  /// Z dimensions with the innermost loop in the nest launched along the
  /// fastest X dimension. For alternative launches, this method may be
  /// overridden to perform a different calculation.
  /// \param f The kernel function into which the tapir loop has been outlined.
  /// \param tl The tapir loop that was outlined.
  /// \param vmap The vmap that maps values from the original host module to
  ///             the device module into which the loop has been outlined.
  virtual void processOutlinedIVs(Function &f, TapirLoopInfo &tl,
                                  ValueToValueMapTy &vmap);

  /// Get the address space, in the device module, for constant global
  /// variables. Assumes the default address space if it is not overridden by
  /// deriving classes.
  virtual unsigned getConstAddrSpace() const { return 0; }

  /// Get the address space, in the device module, for non-constant global
  /// variables. Assumes the default address space if it is not overridden by
  /// deriving classes.
  virtual unsigned getNonConstAddrSpace() const { return 0; }

public:
  /// \param tt The ID of the tapir target that created this outline processor
  /// \param loop The tapir loop being outlined
  /// \param hostM The module containing the tapir loop that is to be outlined
  /// \param devM The module into which the loop will be outlined
  /// \param tto The tapir target options
  /// \param kernelName The name of the function in the device module into which
  ///                   the loop will be outlined
  GPUTTLoopBase(Module &hostM, Module &devM, const TTOptions &tto, TTID tt,
                const TapirLoopInfo &tl, StringRef kernelName);
  virtual ~GPUTTLoopBase() = default;

  /// Setup the loop-control arguments \p lcArgs and loop-control inputs
  /// \p lcInputs for the Tapir loop \p tl.
  void setupLoopControlArgs(TapirLoopInfo *tl, SmallVectorImpl<Value *> &lcArgs,
                            SmallVectorImpl<Value *> &lcInputs) override;

  /// Process the tapir loop \p tl jut before it is outlined. The \p vmap can
  /// be modified here for more control over how the outlining is performed.
  virtual void preProcessTapirLoop(TapirLoopInfo &tl,
                                   ValueToValueMapTy &vmap) override;

  /// Processes the outlined tapir loop \p tl, just after it has been outlined.
  virtual void postProcessOutline(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                                  ValueToValueMapTy &vmap) override;

  /// Processes a call to the function into which the tapir loop \p tl has been
  /// outlined.
  virtual void processOutlinedLoopCall(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                                       DominatorTree &dt) override;
};

} // namespace llvm

#endif // KITSUNE_TARGETS_GPUTT_LOOP_H
