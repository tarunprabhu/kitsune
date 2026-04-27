# Memory Management

This page describes the memory management aspects of Kitsune's **experimental**
programming model. This is constantly evolving, so the strictness or leniency
of the compiler may change without warning.

```{warning}
While we make every effort to keep this documentation up-to-date, any observed
difference in behavior between the compiler and the documentation here likely
means that this documentation is out-of-date - it is _not_ automatically a bug
in the compiler.
```

(memory-management-introduction)=
## Introduction

Since Kitsune allows the same code to be compiled for both CPU and GPU, even
basic memory management tasks such as dynamic memory allocation can be tricky.
If a construct such as a [parallel for loop](extensions-cxx-forall) is
compiled for the GPU, the data on which it operates must be copied to the GPU,
then copied back to the host when it accessed there. To avoid the need for
explicit memory copies, Kitsune requires that any data that is accessed on the
GPU be allocated using Unified Virtual Memory (UVM). This is supported on both
[NVIDIA](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/) and
[AMD](https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.0/how-to/unified_memory.html) GPU's.

By "data", what we really mean is non-scalar data that is accessed by pointer.
Consider the parallel for loop shown below.

```kitc
forall (long i = 0; i < n; ++i)
  c[i] = a[i] + b[i];
```

Here, the arrays `a`, `b`, and `c` are essentially pointers to memory. Those
that are not global variables must have been dynamically allocated using
an appropriate UVM allocator before use on the GPU. However, `n` does not need
to have been allocated using UVM since it is a scalar and is not accessed by
pointer. On the other hand, if `n` were passed by reference to the function
containing the code above, it would have to be allocated using UVM since a
reference in C++ is, from the compiler's perspective, just a non-null pointer.

However, if the code is compiled with a CPU-centric tapir target such as
[opencilk](tapir-targets-opencilk) or [pthreads](tapir-targets-pthreads), there
is no need to allocate memory in any special way - using a standard memory
allocator such as `malloc` is sufficient.

Since an explicit goal of Kitsune's programming model is to ease portability of
code across platforms, programmers using Kitsune should not have to concern
themselves with how to allocate, or free, memory.

While allocating memory using an appropriate memory allocator is necessary for
correct execution[^1], a major consideration is also performance. Some GPU's can
only access GPU memory. Any data on which they operate must be copied from the
host to the GPU first. This memory would have to be copied back from the GPU
before it can be accessed on the host. Kitsune seeks to perform this data
movement efficiently and automatically. For this to work well, Kitsune must know
what memory to copy and when. Keeping track of this is a non-trivial task and
requires co-ordination between the compiler and the runtime.

It is not possible, in the general case, for the compiler to detect which data
is being used in a parallel for loop, then trace it back to where it was
allocated and, based on the tapir target, use an appropriate memory allocator.
Rather than provide a "best-effort" approach, we instead require the programmer
to use custom memory allocation functions and explicitly annotate the pointers
that are expected to be used in code that may be run on a GPU. To this end,
we introduce the notion of "mobile" pointers, which we describe in the
[next section](memory-management-mobile-pointers)

[^1]: On some GPU's, a kernel may crash if it dereferences a pointer to non-global memory that was not allocated using the appropriate memory allocator

(memory-management-mobile-pointers)=
## Mobile pointers

Consider a CPU with an attached GPU. The GPU may have its own memory that is
separate from the CPU (or host) memory.
Each of these [compute units](glossary-compute-unit) can be thought of as having
a distinct [memory space](glossary-memory-space). If the GPU is not allowed to
directly access the CPU's memory, and vice versa, the two memory spaces are
deemed to be _disjoint_.
The same is a true of a parallel computing cluster where CPU's are organized
into "nodes" where all CPU's in a node share memory, but CPU's in different
nodes cannot access each other's memory.

In such cases, a compute unit with memory space `A` that wishes to access data
in a buffer that was allocated in a distinct memory space `B`, must first copy
the data from `B` to `A`, then copy it back to `B` afterwards.

A [mobile buffer](glossary-mobile-buffer) is a region of memory allocated in a
memory space, `A` whose contents may be needed by a compute unit with a disjoint
memory space, `B`.

A [mobile pointer](glossary-mobile-pointer) is a pointer to a
[mobile buffer](glossary-mobile-buffer).

```{important}
All pointers to non-global data used in Kitsune's parallel constructs such as
parallel for loops *must* be mobile pointers.
```

The presence of the ``[[kitsune::mobile]]`` attribute on a pointer type
indicates that the value is a mobile pointer. See the code below for some
examples.

```kitc
void cpy(int *[[kitsune::mobile]] dst, const int *[[kitsune::mobile]] src) {
  ...
}
int *[[kitsune::mobile]] ptr = NULL;
```

In C++, we also provide a template `kitsune::mobile_ptr` class that is a
wrapper around a mobile pointer. This is defined in `kitsune.h`.

```kit++
#include <kitsune.h>

void cpy(kitsune::mobile_ptr<int> &dst, const kitsune::mobile_ptr<int> &src) {
  ...
}
kitsune::mobile_ptr<int> ptr;
```

(memory-management-mobile-pointer-semantics)=
### Semantics

```{warning}
Since this still experimental, we do not yet have a formal semantics for
mobile pointers. Instead, we describe some general rules governing the use of
mobile pointers that are not expected to change. In some cases, these are
rigorously checked by the compiler, which will reject any invalid uses. In
others, the compiler may not raise an error, or even issue a warning, but the
resulting code may crash at runtime.
```
- The `[[kitsune::mobile]]` can be cast away, for example, as shown below

  ```kitc
  float *[[kitsune::mobile]] m = ...;
  float *p = (float*)m;
  ```
  However, this is *NOT* recommended.

- The `[[kitsune::mobile]]` attribute cannot be added to a pointer. For
  instance, the following is explicitly disallowed.

  ```kitc
  float * p = ...;
  float *[[kitsune::mobile]] m = (float *[[kitsune::mobile]]) p; // ERROR
  ```

  This is possible using the `__kitsune_mobile_cast_unsafe` builtin can be used
  instead.

  ```kitc
  float *p = ...;
  float *[[kitsune::mobile]] m = __kitsune_mobile_cast_unsafe(p);
  ```

  This builtin is only intended to be used in Kitsune's runtime. It's use in
  user code is *strongly* discouraged.

- Implicitly casting away the `[[kitsune::mobile]]` attribute is not allowed.
  For instance, the code below will raise a compiler error because the called
  function `f` expected a regular pointer but we are attempting to pass it a
  mobile pointer.

  ```kitc
  void f(void* ptr) {
    ...
  }

  int driver() {
    void *[[kitsune::mobile]] ptr = ...;
    f(ptr);
  }
  ```

  Explicit casts must be used in this case.

  ```kitc
  f((void*) ptr);
  ```

(memory-management-mobile-pointer-allocation)=
### Examples

The [kitsune_mobile_alloc](kitsune_mobile_alloc) builtin must be used to
allocate a mobile buffer. This is intended to be a drop-in replacement for
`malloc` and has exactly the same semantics. The corresponding function to free
a mobile buffer is [kitsune_mobile_free](kitsune_mobile_free). The code below
shows how a buffer containing `n` integers can be allocated, initialized, then
freed.

```kitc
void f(long n) {
  int* [[kitsune::mobile]] buf = kitsune_mobile_alloc(n * sizeof(int));
  forall (long i = 0; i < n; ++i)
    buf[i] = i;
  kitsune_mobile_free(buf);
}
```

#### C++ Only

An alternative to C-style use of mobile buffers is to use the templated
`kitsune::mobile_ptr` class. The example below shows how this class may be used.

```kit++
#include <kitsune.h>

void f(long n) {
  kitsune::mobile_ptr<int> dst(n);
  kitsune::mobile_ptr<int> src;

  src.alloc(n);
  forall (long i = 0; i < n; ++i)
    dst[i] = src[i];

  src.free();
  dst.free();
}
```

Note here that `dst` is allocated by passing the number of elements to allocate
to the constructor. The `alloc` method can be used to defer allocation of the
underlying buffer.

```{note}
Unlike `kitsune_mobile_alloc`, when allocating a mobile buffer using the
kitsune::mobile_ptr<T> class, only the number of elements are required. There is
no need to multiply it by the size of the element.
```

Note also that both buffers were explicitly freed.

```{important}
kitsune::mobile_ptr<T> is *not* a smart pointer. The underlying mobile buffer
must be allocated and freed manually, otherwise memory will leak.
```

Unlike `kitsune_mobile_alloc` and `kitsune_mobile_free` that are compiler
builtins and do not require a header, `kitsune.h` is required in order to use
`kitsune::mobile_ptr<T>`;

#### Experimental

```{warning}
The functions described in this section are only provided for convenience, but
may be removed without notice in the future.
```

In order to aid in porting existing code to use Kitsune, the
`kitsune_mobile_alloc__` builtin can be used to obtain a regular pointer to a
mobile buffer. This is equivalent to the following:

```kitc
void f(long n) {
  void *[[kitsune::mobile]] m = kitsune_mobile_alloc(n);
  void *p = (void*) m;
```

`kitsune_mobile_free__` can be used to free buffers allocated with
`kitsune_mobile_alloc__`. It is an error to use this function to free a buffer
that was *not* allocated by `kitsune_mobile_alloc__`.

The example below shows how these may be used.

```kitc
void f(long n) {
  int* buf = kitsune_mobile_alloc__(n * sizeof(int));
  forall (long i = 0; i < n; ++i)
    buf[i] = i;
  kitsune_mobile_free__(buf);
}
```
