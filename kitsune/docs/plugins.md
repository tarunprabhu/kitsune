## Kitsune plugins

Kitsune currently uses some Clang plugins that provide additional analysis and
code transformation capabilities. These can be found in the kitsune/plugins 
directory. 

The plugins each consist of a frontend and backend component. 

The frontend  typically consists of one or more `ASTConsumer`'s and a
`RecursiveASTVisitor`, all of which operate on the clang AST. They may modify
the AST or simply perform analyses at that level.

The backend component consists of one or more LLVM passes. The passes also
perform analyses and/or transformations. 

Because of the way the plugins are designed, the LLVM passes have access to the 
clang AST which would otherwise not be possible. This allows them to carry out 
language-specific actions guided by the AST which may not be possible if the
language-specific
information needed to enable these actions was lost when lowering to LLVM-IR. 

### Example

An example of such a plugin is the `ConstGlobalVar` plugin. This is used to 
identify global variables that are declared *`const`* in the source code but are 
not considered const in LLVM-IR i.e. the `llvm::GlobalVariable:isConstant()`
method will return *`false`* for the `llvm::GlobalVariable`. 

This is needed
when `extern` global variables are used in C/C++ code. If a file uses an 
*`extern const`* global, the initializer of that global may not be known when 
compiling the file since it will appear in a different translation unit and will
only be resolved at link-time. LLVM, however requires constant 
`GlobalVariable`'s to have an initializer whose value is known at compile time.
Therefore, `extern const` globals are not considered to be constant 
`llvm::GlobalVariable`'s. 

The frontend of the plugin recursively finds all declarations of used global
variables that are declared *`const`* and records them in an object that is
shared with an LLVM pass. The LLVM pass iterates over all `GlobalVariable`'s in
the `Module`, and looks up the corresponding source-level declaration in the
shared object and attaches Kitsune-specific metadata to the `GlobalVariable`
indicating that it is a constant. 

### Architecture

Currently, the plugins are invoked automatically when running kitsune. 
(TODO: It would be much better if these were kept separate from the code and 
run as they were designed to using `-fplugin` and `-fpass-plugin`). 

The sources of all the frontends are combined
into a single shared library. The same is done for all the backend sources. 
This is quite different to the way a Clang plugin is normally built because each
plugin would be compiled into its own shared library. This is done mainly to
avoid having to find all the plugins built as part of Kitsune and loading each
of them separately.

### Adding a new plugin

In order to add a new plugin:

    1. Create a new subdirectory in kitsune/plugins for the plugin. You may 
       organize the frontend and backend of the plugin in any way. Ensure that
       names of any publicly visible functions/methods/types do not
       conflict with those in other plugins. 
       
    2. Create a CMakeLists.txt file for the plugin and in it call 
       `add_kitsune_plugin_frontend_sources` with a list of the source files 
       that comprise the frontend and `add_kitsune_plugin_backend_sources` with
       a list of the source files that comprise the backend. The list files for 
       objects that are meant to be shared between the frontend and the backend
       must be passed to `add_kitsune_plugin_common_sources`. When passing 
       the files, ensure that the full path to the file is passed by prepending
       the file name with `${CMAKE_CURRENT_SOURCE_DIR}`.
       
    3. Register the top-level Plugin class in PluginRegistry.cpp. Follow the 
       instructions in the comments in that file to determine where to insert
       the new plugin relative to the existing plugins.
       
    4. Add the subdirectory containing the sources for your plugin to the 
       PLUGIN_DIRS variable in the CMakeLists.txt file in kitsune/plugins. 
       
Note that a backend is optional for a plugin if the plugin will only carry 
out AST-level transformations. In that case, `add_kitsune_plugin_backend_sources`
and `add_kitsune_plugin_common_sources` need not be called.
