Tapir-Clang
================================

This is an experimental approach to a frontend for tapir. It allows non-nested
concurrency, allowing interleaved tasks. To do so, it addes the `spawn` and
`sync` keywords, only enabled when `-ftapir` is specified. These are used with
identifiers to link spawns with sincs:
  
    spawn statement: "spawn" identifier statement
    sync statement: "sync" identifier

For example:

    spawn f foo();
    bar();
    spawn g baz();
    sync f;
    qux();
    sync g;

    
    
    
