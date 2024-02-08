# The LLVM Compiler Infrastructure

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/llvm/llvm-project/badge)](https://securityscorecards.dev/viewer/?uri=github.com/llvm/llvm-project)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8273/badge)](https://www.bestpractices.dev/projects/8273)
[![libc++](https://github.com/llvm/llvm-project/actions/workflows/libcxx-build-and-test.yaml/badge.svg?branch=main&event=schedule)](https://github.com/llvm/llvm-project/actions/workflows/libcxx-build-and-test.yaml?query=event%3Aschedule)

Welcome to the LLVM project!

This repository contains the source code for LLVM, a toolkit for the
construction of highly optimized compilers, optimizers, and run-time
environments.

The LLVM project has multiple components. The core of the project is
itself called "LLVM". This contains all of the tools, libraries, and header
files needed to process intermediate representations and convert them into
object files. Tools include an assembler, disassembler, bitcode analyzer, and
bitcode optimizer.

C-like languages use the [Clang](https://clang.llvm.org/) frontend. This
component compiles C, C++, Objective-C, and Objective-C++ code into LLVM bitcode
-- and from there into object files, using LLVM.

Other components include:
the [libc++ C++ standard library](https://libcxx.llvm.org),
the [LLD linker](https://lld.llvm.org), and more.

## Getting the Source Code and Building LLVM

Consult the
[Getting Started with LLVM](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)
page for information on building and running LLVM.

For information on how to contribute to the LLVM project, please take a look at
the [Contributing to LLVM](https://llvm.org/docs/Contributing.html) guide.

## Getting in touch

Join the [LLVM Discourse forums](https://discourse.llvm.org/), [Discord
chat](https://discord.gg/xS7Z362),
[LLVM Office Hours](https://llvm.org/docs/GettingInvolved.html#office-hours) or
[Regular sync-ups](https://llvm.org/docs/GettingInvolved.html#online-sync-ups).

The LLVM project has adopted a [code of conduct](https://llvm.org/docs/CodeOfConduct.html) for
participants to all modes of communication within the project.

## Tapir

For the Tapir compiler IR, cite either the
[Tapir conference paper][SchardlMoLe17] at ACM PPoPP 2017 conference
paper or the [Tapir journal paper][SchardlMoLe19] in ACM TOPC 2019.

Tapir conference paper, ACM PPoPP 2017:
> Tao B. Schardl, William S. Moses, and Charles E. Leiserson. 2017.
> Tapir: Embedding Fork-Join Parallelism into LLVM's Intermediate
> Representation. In Proceedings of the 22nd ACM SIGPLAN Symposium
> on Principles and Practice of Parallel Programming (PPoPP '17).
> 249–265. https://doi.org/10.1145/3018743.3018758

BibTeX:
```bibtex
@inproceedings{SchardlMoLe17,
author = {Schardl, Tao B. and Moses, William S. and Leiserson, Charles E.},
title = {Tapir: Embedding Fork-Join Parallelism into LLVM's Intermediate Representation},
year = {2017},
isbn = {9781450344937},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3018743.3018758},
doi = {10.1145/3018743.3018758},
booktitle = {Proceedings of the 22nd ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming},
pages = {249–-265},
numpages = {17},
keywords = {control-flow graph, multicore, tapir, openmp, fork-join parallelism, cilk, optimization, serial semantics, llvm, par- allel computing, compiling},
location = {Austin, Texas, USA},
series = {PPoPP '17}
}
```

Journal article about Tapir, ACM TOPC 2019:
> Tao B. Schardl, William S. Moses, and Charles E. Leiserson. 2019.
> Tapir: Embedding Recursive Fork-join Parallelism into LLVM’s
> Intermediate Representation. ACM Transactions on Parallel Computing 6,
> 4, Article 19 (December 2019), 33 pages. https://doi.org/10.1145/3365655

BibTeX:
```bibtex
@article{SchardlMoLe19,
author = {Schardl, Tao B. and Moses, William S. and Leiserson, Charles E.},
title = {Tapir: Embedding Recursive Fork-Join Parallelism into LLVM’s Intermediate Representation},
year = {2019},
issue_date = {December 2019},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {6},
number = {4},
issn = {2329-4949},
url = {https://doi.org/10.1145/3365655},
doi = {10.1145/3365655},
journal = {ACM Transactions on Parallel Computing},
month = {dec},
articleno = {19},
numpages = {33},
keywords = {compiling, fork-join parallelism, Tapir, control-flow graph, optimization, parallel computing, OpenMP, multicore, Cilk, serial-projection property, LLVM}
}
```

## Acknowledgments

OpenCilk is supported in part by the National Science Foundation,
under grant number CCRI-1925609, and in part by the
[USAF-MIT AI Accelerator](https://aia.mit.edu/), which is sponsored by the
United States Air Force Research Laboratory under Cooperative Agreement
Number FA8750-19-2-1000.

Any opinions, findings, and conclusions or recommendations expressed
in this material are those of the author(s) and should not be
interpreted as representing the official policies or views, either
expressed or implied, of the United states Air Force, the
U.S. Government, or the National Science Foundation.  The
U.S. Government is authorized to reproduce and distribute reprints for
Government purposes notwithstanding any copyright notation herein.

[SchardlLe23]: https://dl.acm.org/doi/10.1145/3572848.3577509
[SchardlMoLe17]: https://dl.acm.org/doi/10.1145/3155284.3018758
[SchardlMoLe19]: https://dl.acm.org/doi/10.1145/3365655

## TODO: 

Add text and acknowledgements for Kitsune
