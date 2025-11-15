#include "kitsune/Core/TTPlugin.h"

using llvm::Module;
using llvm::TapirTarget;
using llvm::TTOptions;

extern "C" ::llvm::TTPluginInfo LLVM_ATTRIBUTE_WEAK llvmGetTTPluginInfo() {
  return {0xbad5, "TTPluginBadAPIVersion", "1.0",
          [](Module &, const TTOptions &) -> TapirTarget * { return nullptr; }};
}
