# kit-sort - Sort the basic blocks in a function in some "reasonable" order

## Synopsis

**kit-sort** [_options_] [_input_]

## Description

Sort the basic blocks in functions in some "reasonable" order. This is usually
reverse postorder since that is closest to "program order", but it may also use
a different, potentially hybrid, ordering. The intention is to make the
control-flow of the function easier to follow since some transformation passes
may order the blocks in a way that obfuscates this.

The output will be written to stdout as human-readable LLVM assembly.

## Options

**--funcs**

: A comma-separated list of functions in the input module whose basic blocks are
  to be sorted. It is an error if any name provided to this option are not
  present in the input. If this option is not provided, all functions in the
  input will be processed.

**--help**

: Print a summary of **kit-sort** command-line options.

**--version**

: Print the version of this program.

## Exit Status

If an error occurs, `kit-sort` exits with a non-zero value. Otherwise, exit
with 0 to indicate success.

## Examples

The most common way to use this is when an LLVM bitcode, or LLVM assembly, file
is passed to `kit-sort`.

```
kit-sort file.bc
kit-sort file.ll
```

The module to process can also be piped into **kit-sort**.

```
cat in.bc | kit-sort
```
