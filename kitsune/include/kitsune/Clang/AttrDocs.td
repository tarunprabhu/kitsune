//==- CAttrDocs.td - Documentation for Kitsune's attributes -*- tablgen -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is expected to be included in clang/Basic/Attr.td and references
// entities that are only defined there. Using this directly is likely to result
// in some error. In almost all cases, clang/Basic/Attr.td should be used.
//
//===----------------------------------------------------------------------===//

// The subclass of the main documentation exists solely to identify
// Kitsune-specific attributes when generating HTML documentation.
// Kitsune-specific attributes *must* use this class. Failure to do so, while
// not incorrect, will result in the attribute not appearing in the generated
// HTML documentation.
class KitDoc : Documentation {}

def TTAttrDoc : KitDoc {
  let Category = DocCatStmt;
  let Content = [{

**Experimental**

.. warning::

   Although this attribute is allowed in the frontend, support for overriding
   the primary tapir target has not been implemented in Kitsune's optimization
   pipeline and will almost certainly result in an error.

This attribute may be attached to a source-level construct that is lowered
to a tapir loop in LLVM-IR. In this case, it overrides the primary tapir target,
that is, the one specified on the command line.

.. code-block:: c++

  [[tapir::target("cuda")]]
  forall(...) {
    ...
  }

Assume that the code above was compiled as shown below

.. code-block:: shell

  kitcc --tapir=opencilk ...

In this case, the primary tapir target is ``opencilk``. All tapir loops in the
code, except the one with the ``tapir::target`` attribute will be lowered to use
the OpenCilk runtime. Here, the loop with the ``tapir::target`` attribute will
be lowered to run on an NVIDIA GPU.

The value of the attribute must be one of the
:ref:`supported tapir targets <tapir-targets-supported>`, *except* for the
``custom`` tapir target which **cannot** be used as a value for this attribute.
  }];
}

def KitLaunchAttrDoc : KitDoc {
  let Category = DocCatStmt;
  let Content = [{

The kitsune launch attributes lets programmers inform the compiler how
to tweak the code generation aspects for launching the GPU kernels for
parallel constructs in the code.  This information is not verified in
only a minor fashion at compile time and then again at runtime.  In
general, it is up to the programmer to communicate the details
correctly in terms of the semantics of the underlying GPU programming
mechanisms. In general, this is intended more for kitsune development
and testing vs. an end-user feature.

- threads_per_block : the number of threads per block to launch the kernel.

.. code-block:: c++

  [[kitsune::launch(THREADS_PER_BLOCK)]]
  forall (...) {
    ...
  }

  }];
}

def KitStmtNameAttrDoc : KitDoc {
  let Category = DocCatStmt;
  let Content = [{
This allows programmers to provide a name for a statement. If this is attached
to a parallel for loop, a a `tapir.loop.name` attribute will be added to the
loop in in LLVM-IR. This is used when emitting diagnostics from Kitsune's
middle-end optimizations. It can also be used by Kitsune's instrumentation
support.

.. code-block:: c++

  [[kitsune::name("some-useful-name")]]
  forall (...) {
    ...
  }

  }];
}

def KitMemAccessAttrDoc : KitDoc {
  let Category = DocCatType;
  let Heading = "_readonly, _readwrite, _writeonly";
  let Content = [{

**Experimental**

The kitsune memory access attributes lets programmers inform the compiler of
how variables are used so the compiler can make better optimization decisions.
The table below describes what each value of this attribute asserts about the
variable to which it is attached.

==============  ===========================================
Attribute       Assertion
==============  ===========================================
``_readonly``   The variable is only read, never written to
``_writeonly``  The variable is only written to, never read
``_readwrite``  The variable is both read and written
==============  ===========================================

In the example below, the attributes are attached to function parameters.

.. code-block:: c++

  void func(_readonly float *a,
            _readwrite float *b,
            _writeonly float *c) {
    ...
  }

Since the attributes are attached to a pointer type, in this case, they assert
that the memory that ``a`` will only be used to read memory, ``b`` will only be
used to both read from and write to memory while ``c`` will only be used to read
memory.

Inconsistent use of these attributes may lead to contradictory assertions about
the underlying memory. For instance, in the code above, ``a`` and ``b`` may
alias. In other words, ``func`` may be called as shown below.

.. code-block:: c++

  func(ptr, ptr, NULL);

Here, the parameters ``a`` and ``b`` point to the same memory. In this case, the
code above implies that the underlying memory is both constant and
non-constant.

Kitsune will make no effort to check for consistent use of these attributes. It
will assume that the programmer has used these attributes correctly. For
instance, Kitsune will not raise an error, or even issue a warning, when
compiling the code below

.. code-block:: c++

  void func(_readonly int* ptr) {
    *ptr = 0xbad;
  }

Here, the programmer has asserted that ``ptr`` is not used to write to memory,
yet it is obvious that the function does so. In such cases, Kitsune may
mis-compile the code by performing more aggressive optimizations.

However, some limited consistency-checking is carried out. It is an error to
attach more than one of these attributes to a single variable. For instance, the
code below will not compile.

.. code-block:: c++

  void func(_readwrite _writeonly int *p) {
     ...
  }

  }];
}

def KitMobileAttrDoc : KitDoc {
  let Category = DocCatType;
  let Heading = "mobile";
  let Content = [{

The mobile attribute should be attached to pointers that may be allocated using
UVM, or that would otherwise point to data that may be moved to a GPU. This is
required when using the GPU tapir targets, but is optional in other cases.
This attribute is incompatible with other "bare" pointers i.e. those that do
not have this attribute. Casting from a bare pointer to a pointer with this
type is not permitted. Explicitly casting away this attribute is allowed.

.. warning::

   The semantics of this attribute may change - as will its "strictness". In
   other words, in the future we may disallow casting away this attribute. We
   may require the types of all entities used in code run on a GPU to have this
   attribute.

The examples below show how the attribute may be used. Note that both the
standard C++ and GNU spellings are allowed. For C, the GNU spelling must be used
if the C standard being used is older than C23. Starting from C23, the standard
C++ spelling is also supported.

.. code-block:: c++

  void f1(int *[[kitsune::mobile]] ptr) {}
  void f2(void *__attribute__((kitsune_mobile)) ptr) {}

  }];
}
