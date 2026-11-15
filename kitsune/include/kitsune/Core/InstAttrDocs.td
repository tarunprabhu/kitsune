//=- InstAttrDocs.td - Docs for Kitsune's instruction attrs -*- tablegen -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Documentation for Kitsune-specific instruction attributes. These are used to
// generate the HTML documentation.
//
//===----------------------------------------------------------------------===//

def KitInstTargetDoc : KitDocAttr {
  let Documentation = [{
This attribute may only be attached to detach, reattach, or sync, instructions.
When present on a matched pair of detach and reattach instructions that are not
in a tapir loop, it indicates that the instructions demarcate a task that is to
be spawned with the tapir target specified in the value of this attribute. When
present on a sync instruction, it indicates that the sync instruction is to be
lowered using the given tapir target.

The presence of this attribute on detach/reattach instructions within a tapir
loop has no effect on the lowering of the loop. However, the target on the
instructions must be the same as that of the tapir loop.
}];
}
