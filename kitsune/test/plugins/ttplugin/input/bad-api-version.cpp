#include "kitsune/Core/TTPlugin.h"

using namespace llvm;

extern "C" ::llvm::TTPluginInfo LLVM_ATTRIBUTE_WEAK llvmGetTTPluginInfo() {
  return {0xbad5,
          "TTPluginBadAPIVersion",
          "1.0",
          [](Module &, const TTOptions &) -> TapirTarget * { return nullptr; },
          /*getCompilerOptions=*/[]() -> TTPlugin::ExtraArgsList { return {}; },
          /*getLinkerOptions=*/[]() -> TTPlugin::ExtraArgsList { return {}; }};
}
