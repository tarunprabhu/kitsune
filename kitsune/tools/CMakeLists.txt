# Add a kitsune tool named NAME. This assumes that the tool will be in a
# subdirectory of the same name in this directory. It also adds the tool to a
# global list of known tools.
function (add_kitsune_tool name)
  add_subdirectory(${name})
  set_property(GLOBAL APPEND PROPERTY KITSUNE_TOOLS ${name})
endfunction ()

add_kitsune_tool(kit-config)
add_kitsune_tool(kit-enc)
add_kitsune_tool(kit-mbc)
add_kitsune_tool(kit-sort)
