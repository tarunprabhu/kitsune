# LLVM Instruction Reference

This is a summary of the LLVM instructions that have been added by Kitsune.
This also includes the instructions that have been added by
[Tapir](faq-difference-kitsune-tapir) since these are critical to Kitsune's
operation. The standard LLVM instructions have been
[documented elsewhere](https://llvm.org/docs/LangRef.html#instruction-reference).

(instructions-kitsune)=
## Kitsune Instructions

Kitsune has not introduced any of its own instructions to LLVM's standard
instruction set.

(instructions-tapir)=
## Tapir Instructions

Since there are limited number of instructions here, we do not further
classify them by kind. The descriptions of the instructions describe what
category they belong to.

(instructions-detach)=
### '`detach`' Instruction

This instruction marks the start of a [tapir task](glossary-tapir-task).

```llvm
detach within %syncreg, label %body, label %continue
```

**Arguments**

```{table}
|||
|:-:|:-:|
| `syncreg` | The sync region with which this instruction is associated |
| `body` | The entry block of the tapir task |
| `continuation` | The continuation block to be executed once the tapir task has completed |
```

**Description**

: This is a terminator instruction. At runtime, this spawns a task starting at
  the `body` block. This is executed in parallel with the `continuation` block.
  Every `detach` instruction must have a corresponding
  [reattach](instructions-reattach) instruction whose operands are `syncreg` and
  `continuation`. The reattach instruction will be part of the tapir task
  spawned by this instruction.


(instructions-reattach)=
### '`reattach`' Instruction

This instruction marks the end of a [tapir task](glossary-tapir-task).

```llvm
reattach within %syncreg, label %continue
```

**Arguments**

```{table}
|||
|:-:|:-:|
| `syncreg` | The sync region with which this instruction is associated |
| `continuation` | The basic block to which control will transfer once the tapir task has completed |
```

**Description**

: This is a terminator instruction. At runtime, it terminates the tapir task
  that it is contained in. Both operands of this instruction, `syncreg` and
  `continuation`, must have been operands to a corresponding
  [detach](instructions-detach) instruction. After terminating the task,
  execution will continue starting at the `continuation` block. Note that a
  continuation block can be entered _exactly once_. If a different tapir task
  with the same continuation block has already begun executing it, the
  reattach instruction will not allow the continuation block to execute.


(instructions-sync)=
### '`sync`' Instruction

Instruction that blocks until all [tasks](glossary-tapir-task) within a
[sync region](glossary-sync-region) have completed.

```llvm
sync within %syncreg, label %exit
```

**Arguments**

```{table}
|||
|:-:|:-:|
| `syncreg` | The sync region with which this instruction is associated |
| `continuation` | The basic block to which control flow will be transferred once this instruction has finished executing |
```

**Description**

: This is a terminator instruction. This instruction will block until all
  tapir tasks spawned - and their descendants - that have been spawned within
  the syncregion `syncreg` have completed. Control will always be transferred
  to the `continuation` block once all tasks have been completed.
