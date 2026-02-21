//===- LoopAttrs.td - Defines loop attributes --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the attributes that may be added to loops.
//
//===----------------------------------------------------------------------===//

// Base class whose specializations specify the type of loop to which an
// attribute can be applied.
class AllowedOnLoopKind;

// The attribute can only be applied to tapir loops.
def TapirLoopsOnly : AllowedOnLoopKind;

// The attribute can only be applied to non-tapir loops.
def NormalLoopsOnly : AllowedOnLoopKind;

// The attribute can be applied to both tapir and non-tapir loops.
def AnyLoop : AllowedOnLoopKind;

// Base class for all Kitsune-specific loop attributes. These are most likely to
// be added to tapir loops, but they may be added to regular loops as well.
// These will, necessarily, be interspersed with other loop "attributes" added
// by LLVM such as "loop.unroll.*" and others that are not specific to tapir
// loops.
//
// When specializing this class, the name of the specialized class will be the
// name of the attribute as it appears in the LoopAttrKind enum. This must be
// in Pascal case i.e. it must start with an uppercase letter and must only
// consist of letters and digits. The name of the attribute as it would appear
// in LLVM-IR metadata is derived from this.
//
// If AllowedOn is set to TapirLoopsOnly, the attributes would appear in
// metadata as shown below.
//
//     .-----------------------------------------.
//     | Name         | Metadata string          |
//     |-----------------------------------------|
//     | Target       | tapir.loop.target        |
//     | PerfectDepth | tapir.loop.perfect.depth |
//     | Grainsize    | tapir.loop.grainsize     |
//     | Dim3         | tapir.loop.dim3          |
//     '-----------------------------------------'
//
// If AllowedOn is set to any value other than TapirLoopsOnly, the attributes
// would appear in metadata as shown below.
//
//     .-----------------------------------.
//     | Name         | Metadata string    |
//     |-----------------------------------|
//     | Target       | loop.target        |
//     | PerfectDepth | loop.perfect.depth |
//     | Grainsize    | loop.grainsize     |
//     | Dim3         | loop.dim3          |
//     '-----------------------------------'
//
class Attr<string ValueTypeParam, AllowedOnLoopKind AllowedOnParam> {
  // If the attribute takes a value, this must be the exact spelling of the
  // C++ type of the argument. If the attribute does not take an argument, this
  // must be an empty string.
  string ValueType = ValueTypeParam;

  // If this is true, the attribute can only be set on tapir loops. Otherwise,
  // it can be set on both tapir loops and regular loops.
  AllowedOnLoopKind AllowedOn = AllowedOnParam;

  // Documentation for this attribute. This is used to generate the HTML
  // documentation.
  code Documentation = [{}];
}

// An attribute that takes an enum argument.
class EnumAttr<string ValueType, AllowedOnLoopKind AllowedOn>
    : Attr<ValueType, AllowedOn>;

// An attribute that does not take an argument.
class FlagAttr<AllowedOnLoopKind AllowedOn> : Attr<"", AllowedOn>;

// An attribute that takes a 32-bit integer argument.
class Int32Attr<AllowedOnLoopKind AllowedOn> : Attr<"int32_t", AllowedOn>;

// An attribute that takes a 64-bit integer argument.
class Int64Attr<AllowedOnLoopKind AllowedOn> : Attr<"int64_t", AllowedOn>;

// An attribute that takes a string argument.
class StrAttr<AllowedOnLoopKind AllowedOn> : Attr<"StringRef", AllowedOn>;

def AutotuneLaunch : FlagAttr<TapirLoopsOnly> {
  let Documentation = [{
If present, this attribute indicates that the runtime should use autotuning to
find optimal kernel launch parameters. This is only used by the 'cuda' and 'hip'
tapir targets.
}];
}

// We name thie Grainsize to ensure that the name of the generated attribute is
// tapir.loop.grainsize. This is backwards compatible with the name of the
// attribute in OpenCilk. We may change this in the future to be more consistent
// with the other attribute names, but that would require changes to most tests
// in llvm/lib/Transforms/Tapir.
def Grainsize : Int32Attr<TapirLoopsOnly> {
  let Documentation = [{
Certain tapir targets may launch one task per iteration of a parallel loop.
The overhead of these task launches and syncs can overwhelm any gains obtained
by running the iterations in parallel. In such cases, it may be beneficial to
reduce the degree of parallelism by strip-mining the loop. When this is done,
a smaller number of parallel tasks will be launched, but each will do extra work
by running some number of iterations serially.

The grain size is the number of iterations that are executed serially in such
cases.

Consider the parallel loop shown on the left below. After strip-mining, it may
resemble the loop on the right. Note the use of the grainsize, `gs` in the
loops on the right.

.. code-block:: c++

  parallel_for(i : {0, n}) {            parallel_for(i : {0, n / gs}) {
                                            for (j : {i * gs, (i + 1) * gs}) {
       ...                                      ...
                                            }
  }                                     }
}];
}

def LoweringEnabled : FlagAttr<TapirLoopsOnly> {
  let Documentation = [{
If this attribute is present in a tapir loop, it will be lowered using the tapir
target specified in the value of this attribute. Tapir loops that do not contain
this attribute will not be lowered directly in the loop spawning pass. This does
not mean that such tapir loops will not be lowered, just that they will not be
passed directly to a tapir target. They will almost certainly be handled when
lowering a parent tapir loop that is passed to a tapir target.
}];
}

def PerfectDepth : Int32Attr<TapirLoopsOnly> {
  let Documentation = [{
The value of the perfect depth attribute indicates the level of the deepest
tapir loop in a tapir loop nest that is perfectly nested relative to the root
i.e. the outermost tapir loop.

.. note::
  A tapir loop nest is a loop nest where the root is a tapir loop. All loops
  between the deepest tapir loop and the root must also be tapir loops.

For instance, in the loops shown below, the perfect depth will be 2.

.. code-block :: c++

  parallel_for (i ...) {                 parallel_for (i ...) {
      parallel_for (j ...) {                 parallel_for (j ...) {
          ...                                    for (k ...) {
      }                                              ...
  }                                              }
                                             }
                                         }

The loop on the left consists of a nest of perfectly nested tapir loops. In this
case, the depth of the loop nest also happens to be 2. On the other hand, the
depth of the loop nest on the right is 3. However, since only the first two
loops in the nest are tapir loops, the perfect depth is 2.

The perfect depth of the loops shown below are also 2.

.. code-block :: c++

  parallel_for (i ...) {                        parallel_for (i ...) {
      parallel_for (j ...) {                        parallel_for (j ...) {
          for (k ...) {                                 stmt;
              parallel_for (l ...) {                    parallel_for (k ...) {
                  ...                                        ...
              }                                         }
          }                                         }
      }                                         }
  }

The perfect depth of that loop is despite all 4 loops in the nest being
perfectly nested since the loop at depth 3 is not a tapir loop. On the right,
the innermost loop is a parallel loop, but it is not perfectly nested relative
to the root which is why the perfect depth is also 2.
}];
}

def PerfectLevel : Int32Attr<TapirLoopsOnly> {
  let Documentation = [{
The value of the perfect level attribute indicates the level of a tapir loop in
a perfect tapir loop nest.

.. note::
  A tapir loop nest is a loop nest where the root is a tapir loop. All loops
  between the deepest tapir loop and the root must also be tapir loops.

In the code shown below, the perfect level of each loop is shown in trailing
comments.

.. code-block::c++

  parallel_for (i ...) {               // Perfect level = 1
      parallel_for (j ...) {           // Perfect level = 2
          for (k ...) {
              parallel_for (l ...) {
                  ...
              }
          }
      }
  }

Here, the innermost tapir loop does not have a perfect level since its parent
is not a tapir loop. As a result, it is not part of the tapir loop nest. On
the other hand, in the loop nest below, the innermost tapir loop does not have
a perfect level since it is not perfectly nested relative to the root of the
tapir loop.

.. code-block::c++

  parallel_for (i ...) {          // Perfect level = 1
      parallel_for (j ...) {      // Perfect level = 2
          stmt;
          parallel_for (k ...) {
               ...
          }
      }
  }

From the examples shown here, it is evident that the perfect level of a tapir
loop must be less than or equal to the perfect depth of the tapir loop nest of
which the tapir loop is a part.
}];
}

def SpawnStrategy : EnumAttr<"TapirSpawnStrategy", TapirLoopsOnly> {
  let Documentation = [{
The spawn strategy attribute mainly determines whether a tapir loop should be
 outlined. At some point, it may be used for something more, but right now, its
use is limited. This attribute is required to be present on all tapir loops.
}];
}

def Target : EnumAttr<"TTID", TapirLoopsOnly> {
  let Documentation = [{
If this attribute is present in a tapir loop, it will be lowered using the tapir
target specified in the attribute's value. This attribute is required to be
present on all tapir loops.
}];
}

def ThreadsPerBlock : Int32Attr<TapirLoopsOnly> {
  let Documentation = [{
If this attribute is present on a tapir loop, its value will be the number of
threads per block used when launching the GPU kernel obtained when the loop is
lowered using one of the GPU-centric tapir targets.
}];
}
