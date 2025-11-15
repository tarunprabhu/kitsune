#include "kitsune/Core/TTPlugin.h"

using llvm::Module;
using llvm::TapirTarget;
using llvm::TTOptions;

extern "C" ::llvm::TTPluginInfo LLVM_ATTRIBUTE_WEAK llvmGetTTPluginInfo() {
  return {LLVM_TTPLUGIN_API_VERSION, "TTPluginBadConstructor", "1.0", nullptr};
}
