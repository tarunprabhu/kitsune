// Every tapir loop attribute is to be defined using one of the
// TAPIR_LOOP_ATTRIBUTE_* macros. The table below summarizes these macros.
//
//   .---------------------------------------------------------------.
//   | Macro                      | Value | Type of value            |
//   | ---------------------------|-------| -------------------------|
//   | TAPIR_LOOP_ATTRIBUTE_ENUM  | Yes   | Enum                     |
//   | TAPIR_LOOP_ATTRIBUTE_FLAG  | No    |                          |
//   | TAPIR_LOOP_ATTRIBUTE_STR   | Yes   | String                   |
//   | TAPIR_LOOP_ATTRIBUTE_UINT  | Yes   | 32-bit unsigned integer  |
//   | TAPIR_LOOP_ATTRIBUTE_ULONG | Yes   | 64-bit unsigned integer  |
//   '---------------------------------------------------------------'
//
// The signatures of the macros are listed below:
//
//     TAPIR_LOOP_ATTRIBUTE_ENUM(NAME, IRNAME, TYPE)
//     TAPIR_LOOP_ATTRIBUTE_FLAG(NAME, IRNAME)
//     TAPIR_LOOP_ATTRIBUTE_STR(NAME, IRNAME)
//     TAPIR_LOOP_ATTRIBUTE_UINT(NAME, IRNAME)
//     TAPIR_LOOP_ATTRIBUTE_ULONG(NAME, IRNAME)
//
// NAME is the name of the attribute. This is the member of the
// TapirLoopAttrKind enum that enumerates all available tapir loop attributes.
// The name must be a valid C++ identifier. It is used to generate the accessor
// functions that will be discussed later.
//
// IRNAME is the string that will appear in the LLVM-IR metadata. It must be a
// string literal that starts with "tapir.loop."
//
// TYPE is the type of the attribute value of the attribute.
//
// In the example below, we define an attribute that takes a 32-bit unsigned
// integer value.
//
//     TAPIR_LOOP_ATTRIBUTE_UINT(Instructions, "tapir.loop.instructions")
//
// This attribute will appear in LLVM-IR metadata as shown below:
//
//     !0 = !{!"tapir.loop.instructions", i32 82}
//
// The NAME field is used to construct the names of the free functions that can
// be used to get, set and query these attributes. For each attribute,
// functions of the form below will be generated:
//
//     bool hasTapirLoop[[NAME]]Atr(const Loop &loop);
//     std::optional<TYPE> getTapirLoop[[NAME]]Attr(const Loop &loop);
//     void addTapirLoop[[NAME]]Attr(Loop &loop, TYPE val);
//     void remoteTapirLoop[[NAME]]Attr(Loop &loop);
//
// Note that in the examples above [[NAME]] should be replaced by the value
// given to the name argument. For instance, the Instructions attribute declared
// above will result in the generation of the following functions:
//
//     bool hasTapirLoopInstructionsAttr(const Loop &loop);
//     std::optional<unsigned> getTapirLoopInstructionsAttr(const Loop &loop);
//     void addTapirLoopInstructionsAttr(Loop &loop, unsigned);
//     void removeTapirLoopInstructionsAttr(Loop &loop);

// If present, this attribute indicates that the runtime should use autotuning
// to try and find the optimal kernel launch parameters (only used when the
// tapir loop is lowered using the 'cuda' or 'hip' tapir targets)
TAPIR_LOOP_ATTRIBUTE_FLAG(AutotuneLaunch, "tapir.loop.autotune.launch")

// The grainsize to use when stripmining (or otherwise lowering) a tapir loop.
TAPIR_LOOP_ATTRIBUTE_UINT(GrainSize, "tapir.loop.grainsize")

// If this attribute is present in a tapir loop, it will be lowered using the
// tapir target specified in the tapir.loop.target attribute. Tapir loops that
// do not contain this attribute will be ignored.
TAPIR_LOOP_ATTRIBUTE_FLAG(LoweringEnabled, "tapir.loop.lowering.enabled")

// The perfect.depth value indicates the level of the deepest tapir loop in a
// tapir loop nest that is perfectly nested relative to the root i.e. the
// outermost tapir loop. A tapir loop nest is a loop nest where the root is a
// tapir loop. All loops between the deepest tapir loop and the root must also
// be tapir loops.
TAPIR_LOOP_ATTRIBUTE_UINT(PerfectDepth, "tapir.loop.perfect.depth")

// The perfect.level attribute value indicates the level of a tapir loop in a
// perfect tapir loop nest. A tapir loop nest is a perfect loop nest where the
// outermost loop is a tapir loop. Roots of tapir loop nests will have a level
// of 1.
TAPIR_LOOP_ATTRIBUTE_UINT(PerfectLevel, "tapir.loop.perfect.level")

// The spawn strategy attribute mainly determines whether a tapir loop should be
// outlined. At some point, it may be used for something more, but right now,
// its use is limited, though it is required.
TAPIR_LOOP_ATTRIBUTE_ENUM(SpawnStrategy, "tapir.loop.spawn.strategy",
                          TapirSpawnStrategy)

// This attribute specifies the tapir target to be used when lowering the loop.
// This may or may not be the same as the primary tapir target.
TAPIR_LOOP_ATTRIBUTE_ENUM(Target, "tapir.loop.target", TTID)

// The threads per block to use when launching a GPU kernel obtained when
// lowering a tapir loop with the 'cuda' or 'hip' tapir targets.
TAPIR_LOOP_ATTRIBUTE_UINT(ThreadsPerBlock, "tapir.loop.threads.per.block")
