/*
 * Copyright (c) 2022 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */

#include "clang/Frontend/FrontendPluginRegistry.h"

#include "ConstGlobalVars/ConstGlobalVarsPlugin.h"

using namespace clang;

// The plugins here should be declared in the order in which they should be
// executed. The plugins should be designed to be independent of one another,
// but in case any of them happens to have a dependence, that should be
// reflected here.
static FrontendPluginRegistry::Add<ConstGlobalVarsPlugin>
    PConstGVS("kitsune-const-global-vars",
              "Identifies top-level variables declared const.");
