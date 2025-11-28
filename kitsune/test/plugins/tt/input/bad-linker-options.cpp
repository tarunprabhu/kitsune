#include "kitsune/Core/TTPlugin.h"

using namespace llvm;

extern "C" ::llvm::TTPluginInfo LLVM_ATTRIBUTE_WEAK llvmGetTTPluginInfo() {
  return {LLVM_TTPLUGIN_API_VERSION,
          "TTPluginBadLinkerOptions",
          "1.0",
          [](Module &, const TTOptions &) -> TapirTarget * { return nullptr; },
          /*getCompilerOptions=*/[]() -> TTPlugin::ExtraArgsList { return {}; },
          /*getLinkerOptions=*/nullptr};
}
