# Sample Pass Plugin

[EmbModulePassPluginDemo.cpp](EmbModulePassPluginDemo.cpp) contains a sample
pass plugin containing an embedded bitcode pass. For more information, consult
[the documentation](../../docs/WritingEmbeddedBitcodePass.md). In particular,
it describes how to build an embedded bitcode pass plugin outside Kitsune.
The [CMakeLists.txt](CMakeLists.txt) file in this directory _cannot_ be used as
a template because it is intended to build this plugin together with Kitsune.
There is dedicated documentation on
[writing a pass plugin](../../docs/WritingPassPlugin.md) as well.
