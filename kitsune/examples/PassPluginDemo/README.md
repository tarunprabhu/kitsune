# Sample Pass Plugin

[PassPluginDemo.cpp](PassPluginDemo.cpp) contains a sample pass plugin.
For more information about pass plugins and writing them, consult the
[documentation](../../docs/WritingPassPlugin.md).  In particular,
it describes how to build an embedded bitcode pass plugin outside Kitsune.
The [CMakeLists.txt](CMakeLists.txt) file in this directory _cannot_ be used as
a template because it is intended to build this plugin together with Kitsune.

The plugin here demonstrates how Kitsune-specific extension points may be used
to schedule custom passes loaded from the pass plugin.
