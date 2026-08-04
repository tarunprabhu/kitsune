//===- GenerateCtorsGPU.h - Generat ctors for GPU tapir targets -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic implementation of ctor/dtor generation for GPU-centric tapir targets.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_TRANSFORMS_GENERATE_CTORS_GPU_H
#define KITSUNE_LIB_TRANSFORMS_GENERATE_CTORS_GPU_H

#include "GenerateCtorsCommon.h"

namespace llvm {

class GlobalVariable;

namespace detail {

/// Base class to generate ctors/dtors for GPU-centric tapir targets. The
/// default implementation should cover most cases for the cuda and hip tapir
/// targets. Aside from the pure virtual methods that must be implemented by
/// subclasses, the two other useful methods that may be overridden are
/// genCtorBeforeDevCodeRegistration() and genCtorAfterDevCodeRegistration().
/// Most calls that setup the runtime should be added to the first of these.
/// genCtorAfterDevCodeRegister is only really necessary if one has to call the
/// runtime to indicate that the device code has been registered, though there
/// may be work that needs to be done after the device code has been registered.
///
/// NOTE: The default implementations of these methods do nothing.
class GenerateCtorGPU : public GenerateCtorBase {
protected:
  const GenerateCtorOptions &genCtorOpts;

protected:
  GenerateCtorGPU(TTID tt, const TTOptions &tto,
                  const GenerateCtorOptions &genCtorOpts);

  std::string getBundleGVName();
  std::string getBundleHandleGVName();

  /// Create a global variable containing the fat binary "bundle". This
  /// consists of the device code and some metadata.
  virtual GlobalVariable *createBundleGV(Module &m, GlobalVariable *devCode);

  /// Create a global variable that will contain the "handle" to the fat binary.
  /// The handle is the value returned by \@llvm.kit.gpu.register.devcode. The
  /// handle is saved into this global and read from there by the global dtor.
  /// and passed to to \@llvm.kit.gpu.unregister.devcode.
  virtual GlobalVariable *createBundleHandleGV(Module &m);

  /// Get the magic number present in the bundle containing the device code.
  virtual int getBundleMagic() const = 0;

  /// Get the version of the bundle.
  virtual int getBundleVersion() const = 0;

  /// Get the object-file section in which the bundle must be present.
  virtual StringRef getBundleSection() const = 0;

  /// Register non-constant global variables that are present in the device
  /// module, \p devM.
  virtual void registerNonConstGlobals(IRBuilder<> &builder,
                                       Value *bundleHandle, const Module &devM);

  /// Register the device code, and all non-const global variables in the device
  /// code.
  virtual void genCtorDevCodeRegistration(IRBuilder<> &builder,
                                          GlobalVariable *gBundle,
                                          GlobalVariable *gBundleHandle,
                                          const Module &devM);

  /// Add additional code to the ctor before the device code is registered. The
  /// default genCtor implementation does a lot of the work that is common to
  /// the GPU-centric tapir targets. But some targets may have to add custom
  /// code to the ctor. This callback affords them the chance to do so. This is
  /// called after the common work is done, but before the device code is
  /// registered. Essentially, the structure of the ctor is as shown below:
  ///
  ///     common-code
  ///     genCtorBeforeDevCodeRegistration()
  ///     genCtorDevCodeRegistration()
  ///     genCtorAfterDevCodeRegistration()
  ///
  virtual void genCtorBeforeDevCodeRegistration(IRBuilder<> &builder);

  /// Add additional code to the ctor after the device code and non-const
  /// globals have been registered. Essentially, the structure of the ctor is as
  /// shown below:
  ///
  ///     common-code
  ///     genCtorBeforeDevCodeRegistration()
  ///     genCtorDevCodeRegistration()
  ///     genCtorAfterDevCodeRegistration()
  ///
  virtual void genCtorAfterDevCodeRegistration(IRBuilder<> &builder,
                                               GlobalVariable *gBundleHandle,
                                               const Module &devM);

  /// Generate the ctor. It is the caller's responsibility to append the
  /// returned function to \@llvm.global_ctors.
  virtual Function *genCtor(Module &m, GlobalVariable *gBundle,
                            GlobalVariable *gBundleHandle, const Module &devM);

  /// Generate the dtor. It is the caller's responsibility to append the
  /// returned function to \@llvm.global_dtors.
  virtual Function *genDtor(Module &m, GlobalVariable *gBundleHandle);

public:
  virtual ~GenerateCtorGPU() = default;
  virtual void run(Module &m) override;
};

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_GENERATE_CTORS_GPU_H
