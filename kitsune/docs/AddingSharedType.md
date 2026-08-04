# Adding a type shared by the compiler and runtime

In some cases, Kitsune computes some properties at compile-time that are used by
the runtime. In some cases, this data may be augmented with runtime information
before use. In such cases, the compiler generally creates some constant data in
the generated executable that is then read by the runtime. Essentially, this
requires a type that is shared by both the compiler and the runtime since it is
the only way to guarantee that the layout of the data is consistent. Creating
such types requires some care since the compiler and the runtime could
potentially be compiled with different compilers, which may affect how any
non-scalar types are laid out in memory.

```{note}
For instance the runtime may, be compiled with Kitsune itself, while Kitsune is
compiled with, say GCC. This has been the case in the past, though both are
compiled with the same compiler currently. But that may well change in the
future.
```

## Requirements

These types *MUST* be POD (Plain Old Data). In C++-speak, they must be
"trivial" and "standard layout".
[^1] provides a readable description of these properties.
[^2] and [^3] are the canonical references, but the slightly more useful ones
are [^4] and [^5]. The [Examples](#examples) section provides examples of such
types.


Care must be taken to ensure that these types are value-initialized before use.
This is generally only a concern in the compiler since the runtime is only
expected to consume these types.

## Examples

TODO: Provide some examples

## References

[^1]: [Trivial, standard-layout, POD, and literal types](https://learn.microsoft.com/en-us/cpp/cpp/trivial-standard-layout-and-pod-types)
[^2]: [C++ Named Requirements: TrivialType](https://en.cppreference.com/cpp/named_req/TrivialType)
[^3]: [C++ Named Requirements: StandardLayoutType](https://en.cppreference.com/cpp/named_req/StandardLayoutType)
[^4]: [Trivial class](https://en.cppreference.com/cpp/language/classes#Trivial_classs)
[^5]: [Standard-layout](https://en.cppreference.com/cpp/language/data_members#Standard-layout)
