#include "kitsune/Core/TTPlugin.h"

using llvm::Module;
using llvm::TapirTarget;
using llvm::TTOptions;

// The expected plugin entry point is llvmGetTTPluginInfo. The function below
// does not have the correct name.
extern "C" ::llvm::TTPluginInfo LLVM_ATTRIBUTE_WEAK llvmGetTTPlugin() {
  return {LLVM_TTPLUGIN_API_VERSION, "TTPluginBadEntryPoint", "1.0",
          [](Module &, const TTOptions &) -> TapirTarget * { return nullptr; }};
}
