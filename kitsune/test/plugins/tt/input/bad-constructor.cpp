#include "kitsune/Core/TTPlugin.h"

using namespace llvm;

extern "C" ::llvm::TTPluginInfo LLVM_ATTRIBUTE_WEAK llvmGetTTPluginInfo() {
  return {LLVM_TTPLUGIN_API_VERSION,
          "TTPluginBadConstructor",
          "1.0",
          /*makeTapirTarget=*/nullptr,
          /*getCompilerOptions=*/[]() -> TTPlugin::ExtraArgsList { return {}; },
          /*getLinkerOptions=*/[]() -> TTPlugin::ExtraArgsList { return {}; }};
}
