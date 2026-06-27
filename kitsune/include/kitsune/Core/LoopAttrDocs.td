//==- LoopAttrDocs.td - Docs for Kitsune's loop attributes -*- tablegen -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Documentation for Kitsune-specific loop attributes. These are used to
// generate the HTML documentation.
//
//===----------------------------------------------------------------------===//

def KitLoopAutotuneLaunchDoc : KitDocAttr {
  let Documentation = [{
If present, this attribute indicates that the runtime should use autotuning to
find optimal kernel launch parameters. This is only used by the
:ref:`cuda <tapir-targets-cuda>` and :ref:`hip <tapir-targets-hip>`
tapir targets.
}];
}

def KitLoopGrainsizeDoc : KitDocAttr {
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

def KitLoopLoweringEnabledDoc : KitDocAttr {
  let Documentation = [{
If this attribute is present in a tapir loop, it will be lowered using the tapir
target specified in the value of this attribute. Tapir loops that do not contain
this attribute will not be lowered directly in the loop spawning pass. This does
not mean that such tapir loops will not be lowered, just that they will not be
passed directly to a tapir target. They will almost certainly be handled when
lowering a parent tapir loop that is passed to a tapir target.
}];
}

def KitLoopNameDoc : KitDocAttr {
  let Documentation = [{
A human-readable name for the loop. This is useful in debugging and diagnostics
when debug information is not available.
}];
}

def KitLoopPerfectDepthDoc : KitDocAttr {
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

def KitLoopPerfectLevelDoc : KitDocAttr {
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

def KitLoopPreparedDoc : KitDocAttr {
  let Documentation = [{
The presence of this attribute indicates that the tapir loop has been
transformed for parallel execution. This is used to ensure that a tapir loop is
not processed twice when doing so may result in severely degraded performance,
or even incorrect execution.
}];
}

def KitLoopReductionDoc : KitDocAttr {
  let Documentation = [{
The presence of this attribute indicates that the loop performs one or more
reductions. Although not strictly required, this attribute is only really useful
(and used) when the loop is a tapir loop.
}];
}

def KitLoopSerializedDoc : KitDocAttr {
   let Documentation = [{
The presence of this attribute indicates that the loop on which it is present
is a tapir loop that was serialized.
}];
}

def KitLoopSpawnStrategyDoc : KitDocAttr {
  let Documentation = [{
The spawn strategy attribute mainly determines whether a tapir loop should be
outlined. At some point, it may be used for something more, but right now, its
use is limited. This attribute is required to be present on all tapir loops.
}];
}

def KitLoopTargetDoc : KitDocAttr {
  let Documentation = [{
If this attribute is present in a tapir loop, it will be lowered using the tapir
target specified in the attribute's value.

.. important::

  This attribute is *required* on all tapir loops. Even if a loop contains
  detach and reattach instructions, and has the expected tapir loop structure
  but does not have this attribute, it will not be recognized as a tapir loop.

.. warning::

  If this attribute is present on a loop that is not actually a tapir loop - for
  instance because it does *NOT* contain detach and reattach instructions - that
  loop may be misidentified as being a tapir loop. This can result in subtle
  bugs.
}];
}

def KitLoopThreadsPerBlockDoc : KitDocAttr {
  let Documentation = [{
If this attribute is present on a tapir loop, its value will be the number of
threads per block used when launching the GPU kernel obtained when the loop is
lowered using one of the GPU-centric tapir targets.
}];
}
