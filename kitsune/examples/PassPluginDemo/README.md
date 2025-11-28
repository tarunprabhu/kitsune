# Sample Pass Plugin

[`PassPluginDemo.cpp`](PassPluginDemo.cpp) contains a sample pass plugin
intended to be loaded from a dynamic shared object. For more information about
writing and building pass plugins outside Kitsune, consult the
[documentation](../../docs/WritingPassPlugin.md), in particular for details
about how to build the plugin outside Kitsune.

The plugin here is mainly intended to demonstrate how Kitsune-specific
extension points may be used to schedule custom passes loaded from the pass
plugin. For more information about writing passes and pass plugins, consult
LLVM's [documentation](../../../llvm/docs/WritingAnLLVMNewPMPass.rst).
