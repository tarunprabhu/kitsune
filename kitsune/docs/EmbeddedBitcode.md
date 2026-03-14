# Embedded Bitcode

When compiling with a GPU-centric [tapir target](glossary-tapir-target),
currently [cuda](tapir-targets-cuda) and [hip](tapir-targets-hip), Kitsune must
generate code for two different architectures - the [host](glossary-host) and
the [device](glossary-device). Since all machine code generation in LLVM takes
place at the granularity of an [LLVM module](glossary-module), at some point
during the compilation, will need at least two modules, the
[host module](glossary-host-module) that will be compiled for the primary
execution unit, typically the CPU and one or more
[device modules](glossary-device-module) that will be compiled for accelerators,
typically GPUs. LLVM's design presumes the existence of exactly one module.
This is reflected, in the design of LLVM's pass managers and the passes
themselves. LLVM's tools such as
[opt](https://llvm.org/docs/CommandGuide/opt.html) and
[llc](https://llvm.org/docs/CommandGuide/llc.html)
are also designed with this assumption. To make effective use of LLVM's
infrastructure and tools, therefore, Kitsune embeds the device modules in the
host module. This document describes the design of this part of Kitsune.

## Requirements

<!--
    XXX: The bullet points for the lists here intentionally alternate. It
    seems ot be the only way to convince sphinx to render this as a loose
    list. Yes, it is utterly ridiculous! There is probably some other stupid
    bug somewhere, but I couldn't be bothered to try and track it down.
-->
* **Minimize changes to LLVM's infrastructure**: Since LLVM's internal
  architecture is designed to operate on a single module, adding support for
  an optional [device module](glossary-device-module) would require extensive
  changes to LLVM's core code base [^1]. This level of engineering effort was
  not feasible at the time that Kitsune was initially developed.


- **Maintain the single-module constraint**: LLVM's tools, particularly `opt`
  and `llc` are designed to operate on a single module. When invoking these
  tools, the user provides a single LLVM module on which to run some
  middle/back-end passes. Retaining this interface would be preferable - in
  other words, the usage of these tools should not have to change appreciably
  [when using Kitsune's extensions](LLVMTools.md). For this to work, the
  "device" module would have to be associated with the source module in some
  way.


* **Avoid associating host and device modules via the filesystem**: One way
  that multiple modules could be associated is via their file names or locations
  in the filesystem. For instance, if `hello.ll` were the bitcode file
  corresponding to the [host module](glossary-host-module), the device module
  could be expected to be named `hello.device.ll` and present in the same
  directory as `hello.ll`. Such approaches are inherently extremely fragile and
  provide poor user experience - both for end users and Kitsune's developers.
  The only way around this would be to "embed" the device module in the host
  module in some way.


- **Avoid changes to LLVM's serialization formats**: Since the device module
  will now be embedded in the host module, we would have to either change both
  the machine-readable [LLVM bitcode](glossary-bitcode) and human-readable
  [LLVM assembly](glossary-llvm-assembly) formats, or find a way to embed the
  device module within the existing features provided by LLVM.


* **Support more than one embedded device module**: Kitsune may well be
  extended to support compiling for more than one device module. This could
  be the case when compiling for a system containing multiple GPU's with
  different architectures that are intended to be used simultaneously.

[^1]: In principle, more than one optional embedded module would have to be supported since we may be compling for a system containing multiple GPU's with different architectures.

## Embedded Bitcode Design

Now that the [requirements](#requirements) have been outlined, we describe how
the embedded bitcode is implemented.

- The device modules are serialized to binary [LLVM bitcode](glossary-bitcode).
  These are converted to [Base64](https://en.wikipedia.org/wiki/Base64) using
  the [encoder](https://llvm.org/doxygen/Base64_8h_source.html) provided by
  LLVM.

* These Base64 encoded device modules are saved as the initializers of global
  variables in the host module. The Base64 encoding ensures that, when the
  host module is serialized to human-readable
  [LLVM assembly](glossary-llvm-assembly), the initializers of the globals
  only contain printable characters.

- These global variables are currently named, but the names are not
  significant. In the future, unnamed globals may be used.

* These global variables have the [kit_bc](llvm-attr-kit-bc) attribute
  indicating that the initializer consists of a serialized device module. It
  also has the [kit_tt](llvm-attr-kit-tt) attribute whose value is the
  integer representation of the tapir target that created the module. If the
  attribute indicates that the global variable was created by the
  [cuda](tapir-targets-cuda) tapir target, the device module will be compiled
  for an NVIDIA GPU. If the attribute indicates that the global variable was
  created by the [hip](tapir-targets-hip) tapir target, the device module will
  be compiled for an AMD GPU.

```{note}
**Future changes**: These globals are not added to a specific section, but we
may do so in the future, especially if we choose to use
[Just-In-Time](https://en.wikipedia.org/wiki/Just-in-time_compilation) (JIT)
compilation in Kitsune's runtime. In this case, these globals will be retained
in the final machine code. Currently, they are deleted by the
[kit-cfgb](passes-kit-cgfb) pass.
```

The code below is an example of a host module containing embedded bitcode.
In this case, the bitcode was created by the `cuda` tapir target.

```kitll
target triple = "x86_64-pc-linux-gnu"

@.kit.emb.bc = unnamed_addr constant [1524 x i8] c"BC\C0\DE5\14\00\00\05\00\00\00b\0C0$JY\BE\A6]\FB\B5\7F\0BQ\80L\01\00\00\00!\0C\00\00`\01\00\00\0B\02!\00\02\00\00\00\22\00\00\00\07\81#\91A\C8\04I\06\1029\92\01\84\0C%\05\08\19\1E\04\8Bb\80\08E\02B\92\0BBD\102\148\08\18K\0A2\22\88H\B0d!C\86\88\04G\1C2B$q\C8\08\11$)@\86\8C\10K\012d\84\08\92\1C #B\88\E5\00\19\11B\04\19*(*\90Q\\ #\B9@\86\0C\19\C3\07\CB\15\19\22\8C\8C$\07\192b,9\C8\90\11#\C8\12\88\0E\1D:dDG\C8\10\22CF\02\19\1A\00\89 \00\00\09\00\00\00\22f\04\10\B2B\82\89\10RB\82\89\90q\C2PH\0A\09&B\C6\05B\22&\08\84\81\809\020\00\00\1A!L\0E\0F\DE\9CNN\BB}\12\1B\04\8A\96\05\00\00d\81\00\00\00\05\00\00\002\1E\98\04\19\11L\90\8C\09&G\C6\04C\AA\08\00\00\00\B1\18\00\00\CB\00\00\003\08\80\1C\C4\E1\1Cf\14\01=\88C8\84\C3\8CB\80\07yx\07s\98q\0C\E6\00\0F\ED\10\0E\F4\80\0E3\0CB\1E\C2\C1\1D\CE\A1\1Cf0\05=\88C8\84\83\1B\CC\03=\C8C=\8C\03=\CCx\8Ctp\07{\08\07yH\87pp\07zp\03vx\87p \87\19\CC\11\0E\EC\90\0E\E10\0Fn0\0F\E3\F0\0E\F0P\0E3\10\C4\1D\DE!\1C\D8!\1D\C2a\1Ef0\89;\BC\83;\D0C9\B4\03<\BC\83<\84\03;\CC\F0\14v`\07{h\077h\87rh\077\80\87p\90\87p`\07v(\07v\F8\05vx\87w\80\87_\08\87q\18\87r\98\87y\98\81,\EE\F0\0E\EE\E0\0E\F5\C0\0E\EC0\03b\C8\A1\1C\E4\A1\1C\CC\A1\1C\E4\A1\1C\DCa\1C\CA!\1C\C4\81\1D\CAa\06\D6\90C9\C8C9\98C9\C8C9\B8\C38\94C8\88\03;\94\C3/\BC\83<\FC\82;\D4\03;\B0\C3\0C\C7i\87pX\87rp\83th\07x`\87t\18\87t\A0\87\19\CES\0F\EE\00\0F\F2P\0E\E4\90\0E\E3@\0F\E1 \0E\ECP\0E3 (\1D\DC\C1\1E\C2A\1E\D2!\1C\DC\81\1E\DC\E0\1C\E4\E1\1D\EA\01\1Ef\18Q8\B0C:\9C\83;\CCP$v`\07{h\077`\87wx\07x\98QL\F4\90\0F\F0P\0E3\1Ej\1E\CAa\1C\E8!\1D\DE\C1\1D~\01\1E\E4\A1\1C\CC!\1D\F0a\06T\85\838\CC\C3;\B0C=\D0C9\FC\C2<\E4C;\88\C3;\B0\C3\8C\C5\0A\87y\98\87w\18\87t\08\07z(\07r\98\81\\\E3\10\0E\EC\C0\0E\E5P\0E\F30#\C1\D2A\1E\E4\E1\17\D8\E1\1D\DE\01\1EfH\19;\B0\83=\B4\83\1B\84\C38\8CC9\CC\C3<\B8\C19\C8\C3;\D4\03<\CCH\B4q\08\07v`\07q\08\87qX\87\19\DB\C6\0E\EC`\0F\ED\E0\06\F0 \0F\E50\0F\E5 \0F\F6P\0En\10\0E\E30\0E\E50\0F\F3\E0\06\E9\E0\0E\E4P\0E\F80#\E2\ECa\1C\C2\81\1D\D8\E1\17\EC!\1D\E6!\1D\C4!\1D\D8!\1D\E8!\1Ff \9D;\BCC=\B8\039\94\839\CCX\BCpp\07wx\07z\08\07zH\87wp\87\19\CB\E7\0E\EF0\0F\E1\E0\0E\E9@\0F\E9\A0\0F\E50\C3\01\03s\A8\07w\18\87_\98\87pp\87t\A0\87t\D0\87r\98\81\84A9\E0\C38\B0C=\90C9\CC@\C4\A0\1D\CA\A1\1D\E0A\1E\DE\C1\1Cf$c0\0E\E1\C0\0E\EC0\0F\E9@\0F\E50C!\83u\18\07sH\87_\A0\87|\80\87r\98\B1\94\01<\8C\C3<\94\C38\D0C:\BC\83;\CC\C3\8C\C5\0CH!\15Ba\1E\E6!\1D\CE\C1\1DR\81\14fLg0\0E\EF \0F\EF\E0\06\EFP\0F\F40\0F\E9@\0E\E5\E0\06\E6 \0F\E1\D0\0E\E50\A3@\83vh\07y\08\87\19R\1A\B8\C3;\84\03;\A4C8\CC\83\1B\84\039\90\83<\CC\03<\84\C38\94\03\00\00\00\00y \00\00\19\00\00\00\92\1EH C\88\0C\19\09dd\C8\C9 \81\8C\042FFF\13\81B\A0\90\F1\C4\C8\089B\86\8Cb@,\00\00\00\07\00\00\00<stdin>\00#\08\010C \CC\10\042\12\98\A0\DC\D6\D2\E8\E6\EA\DC\CA\\\C8\CA\EC\D2\C6\CA\\\DA\DE\C8\EA\D8\CA\\\CC\D8\C2\CE\E6F\11\84\01\00\00\00\A9\18\00\00-\00\00\00\0B\0Ar(\87w\80\07zXp\98C=\B8\C38\B0C9\D0\C3\82\E6\1C\C6\A1\0D\E8A\1E\C2\C1\1D\E6!\1D\E8!\1D\DE\C1\1D\164\E3`\0E\E7P\0F\E1 \0F\E4@\0F\E1 \0F\E7P\0E\F4\B0\80\81\07y(\87p`\07vx\87q\08\07z(\07rXp\9C\C38\B4\01;\A4\83=\94\C3\02k\1C\D8!\1C\DC\E1\1C\DC \1C\E4a\1C\DC \1C\E8\81\1E\C2a\1C\D0\A1\1C\C8a\1C\C2\81\1D\D8a\C1\01\0F\F4 \0F\E1P\0F\F4\80\0E\0B\88u\18\07sH\87\05\CF8\BC\83;\D8C9\C8\C39\94\83;\8CC9\8C\03=\C8\03;\00\00\00\00\D1\10\00\00\06\00\00\00\07\CC<\A4\83;\9C\03;\94\03=\A0\83<\94C8\90\C3\01\00\00\00q \00\00\02\00\00\002\0E\10\22\04\00\00\00\00\00\00\00]\0C\00\00\11\00\00\00\12\03\94v\00\00\00\0021.1.3 a9a836d75454d0e801a53954ccbbe3a07732d70e<stdin>\00\00\00\00\00\00" #0

attributes #0 = { kit_bc kit_tt(2) }
```

The code below shows a module containing embedded bitcode generated by
two different tapir targets. The global variables are unnamed to emphasize that
it is legal for a host module to contain embedded bitcode in unnamed globals.

```kitll
target triple = "x86_64-pc-linux-gnu"

@1 = unnamed_addr constant [1524 x i8] c"BC\C0\DE5\14\00\00\05\00\00\00b\0C0$JY\BE\A6]\FB\B5\7F\0BQ\80L\01\00\00\00!\0C\00\00`\01\00\00\0B\02!\00\02\00\00\00\22\00\00\00\07\81#\91A\C8\04I\06\1029\92\01\84\0C%\05\08\19\1E\04\8Bb\80\08E\02B\92\0BBD\102\148\08\18K\0A2\22\88H\B0d!C\86\88\04G\1C2B$q\C8\08\11$)@\86\8C\10K\012d\84\08\92\1C #B\88\E5\00\19\11B\04\19*(*\90Q\\ #\B9@\86\0C\19\C3\07\CB\15\19\22\8C\8C$\07\192b,9\C8\90\11#\C8\12\88\0E\1D:dDG\C8\10\22CF\02\19\1A\00\89 \00\00\09\00\00\00\22f\04\10\B2B\82\89\10RB\82\89\90q\C2PH\0A\09&B\C6\05B\22&\08\84\81\809\020\00\00\1A!L\0E\0F\DE\9CNN\BB}\12\1B\04\8A\96\05\00\00d\81\00\00\00\05\00\00\002\1E\98\04\19\11L\90\8C\09&G\C6\04C\AA\08\00\00\00\B1\18\00\00\CB\00\00\003\08\80\1C\C4\E1\1Cf\14\01=\88C8\84\C3\8CB\80\07yx\07s\98q\0C\E6\00\0F\ED\10\0E\F4\80\0E3\0CB\1E\C2\C1\1D\CE\A1\1Cf0\05=\88C8\84\83\1B\CC\03=\C8C=\8C\03=\CCx\8Ctp\07{\08\07yH\87pp\07zp\03vx\87p \87\19\CC\11\0E\EC\90\0E\E10\0Fn0\0F\E3\F0\0E\F0P\0E3\10\C4\1D\DE!\1C\D8!\1D\C2a\1Ef0\89;\BC\83;\D0C9\B4\03<\BC\83<\84\03;\CC\F0\14v`\07{h\077h\87rh\077\80\87p\90\87p`\07v(\07v\F8\05vx\87w\80\87_\08\87q\18\87r\98\87y\98\81,\EE\F0\0E\EE\E0\0E\F5\C0\0E\EC0\03b\C8\A1\1C\E4\A1\1C\CC\A1\1C\E4\A1\1C\DCa\1C\CA!\1C\C4\81\1D\CAa\06\D6\90C9\C8C9\98C9\C8C9\B8\C38\94C8\88\03;\94\C3/\BC\83<\FC\82;\D4\03;\B0\C3\0C\C7i\87pX\87rp\83th\07x`\87t\18\87t\A0\87\19\CES\0F\EE\00\0F\F2P\0E\E4\90\0E\E3@\0F\E1 \0E\ECP\0E3 (\1D\DC\C1\1E\C2A\1E\D2!\1C\DC\81\1E\DC\E0\1C\E4\E1\1D\EA\01\1Ef\18Q8\B0C:\9C\83;\CCP$v`\07{h\077`\87wx\07x\98QL\F4\90\0F\F0P\0E3\1Ej\1E\CAa\1C\E8!\1D\DE\C1\1D~\01\1E\E4\A1\1C\CC!\1D\F0a\06T\85\838\CC\C3;\B0C=\D0C9\FC\C2<\E4C;\88\C3;\B0\C3\8C\C5\0A\87y\98\87w\18\87t\08\07z(\07r\98\81\\\E3\10\0E\EC\C0\0E\E5P\0E\F30#\C1\D2A\1E\E4\E1\17\D8\E1\1D\DE\01\1EfH\19;\B0\83=\B4\83\1B\84\C38\8CC9\CC\C3<\B8\C19\C8\C3;\D4\03<\CCH\B4q\08\07v`\07q\08\87qX\87\19\DB\C6\0E\EC`\0F\ED\E0\06\F0 \0F\E50\0F\E5 \0F\F6P\0En\10\0E\E30\0E\E50\0F\F3\E0\06\E9\E0\0E\E4P\0E\F80#\E2\ECa\1C\C2\81\1D\D8\E1\17\EC!\1D\E6!\1D\C4!\1D\D8!\1D\E8!\1Ff \9D;\BCC=\B8\039\94\839\CCX\BCpp\07wx\07z\08\07zH\87wp\87\19\CB\E7\0E\EF0\0F\E1\E0\0E\E9@\0F\E9\A0\0F\E50\C3\01\03s\A8\07w\18\87_\98\87pp\87t\A0\87t\D0\87r\98\81\84A9\E0\C38\B0C=\90C9\CC@\C4\A0\1D\CA\A1\1D\E0A\1E\DE\C1\1Cf$c0\0E\E1\C0\0E\EC0\0F\E9@\0F\E50C!\83u\18\07sH\87_\A0\87|\80\87r\98\B1\94\01<\8C\C3<\94\C38\D0C:\BC\83;\CC\C3\8C\C5\0CH!\15Ba\1E\E6!\1D\CE\C1\1DR\81\14fLg0\0E\EF \0F\EF\E0\06\EFP\0F\F40\0F\E9@\0E\E5\E0\06\E6 \0F\E1\D0\0E\E50\A3@\83vh\07y\08\87\19R\1A\B8\C3;\84\03;\A4C8\CC\83\1B\84\039\90\83<\CC\03<\84\C38\94\03\00\00\00\00y \00\00\19\00\00\00\92\1EH C\88\0C\19\09dd\C8\C9 \81\8C\042FFF\13\81B\A0\90\F1\C4\C8\089B\86\8Cb@,\00\00\00\07\00\00\00<stdin>\00#\08\010C \CC\10\042\12\98\A0\DC\D6\D2\E8\E6\EA\DC\CA\\\C8\CA\EC\D2\C6\CA\\\DA\DE\C8\EA\D8\CA\\\CC\D8\C2\CE\E6F\11\84\01\00\00\00\A9\18\00\00-\00\00\00\0B\0Ar(\87w\80\07zXp\98C=\B8\C38\B0C9\D0\C3\82\E6\1C\C6\A1\0D\E8A\1E\C2\C1\1D\E6!\1D\E8!\1D\DE\C1\1D\164\E3`\0E\E7P\0F\E1 \0F\E4@\0F\E1 \0F\E7P\0E\F4\B0\80\81\07y(\87p`\07vx\87q\08\07z(\07rXp\9C\C38\B4\01;\A4\83=\94\C3\02k\1C\D8!\1C\DC\E1\1C\DC \1C\E4a\1C\DC \1C\E8\81\1E\C2a\1C\D0\A1\1C\C8a\1C\C2\81\1D\D8a\C1\01\0F\F4 \0F\E1P\0F\F4\80\0E\0B\88u\18\07sH\87\05\CF8\BC\83;\D8C9\C8\C39\94\83;\8CC9\8C\03=\C8\03;\00\00\00\00\D1\10\00\00\06\00\00\00\07\CC<\A4\83;\9C\03;\94\03=\A0\83<\94C8\90\C3\01\00\00\00q \00\00\02\00\00\002\0E\10\22\04\00\00\00\00\00\00\00]\0C\00\00\11\00\00\00\12\03\94v\00\00\00\0021.1.3 a9a836d75454d0e801a53954ccbbe3a07732d70e<stdin>\00\00\00\00\00\00" #0
@2 = unnamed_addr constant [1524 x i8] c"BC\C0\DE5\14\00\00\05\00\00\00b\0C0$JY\BE\A6]\FB\B5\7F\0BQ\80L\01\00\00\00!\0C\00\00`\01\00\00\0B\02!\00\02\00\00\00\22\00\00\00\07\81#\91A\C8\04I\06\1029\92\01\84\0C%\05\08\19\1E\04\8Bb\80\08E\02B\92\0BBD\102\148\08\18K\0A2\22\88H\B0d!C\86\88\04G\1C2B$q\C8\08\11$)@\86\8C\10K\012d\84\08\92\1C #B\88\E5\00\19\11B\04\19*(*\90Q\\ #\B9@\86\0C\19\C3\07\CB\15\19\22\8C\8C$\07\192b,9\C8\90\11#\C8\12\88\0E\1D:dDG\C8\10\22CF\02\19\1A\00\89 \00\00\09\00\00\00\22f\04\10\B2B\82\89\10RB\82\89\90q\C2PH\0A\09&B\C6\05B\22&\08\84\81\809\020\00\00\1A!L\0E\0F\DE\9CNN\BB}\12\1B\04\8A\96\05\00\00d\81\00\00\00\05\00\00\002\1E\98\04\19\11L\90\8C\09&G\C6\04C\AA\10\00\00\00\B1\18\00\00\CB\00\00\003\08\80\1C\C4\E1\1Cf\14\01=\88C8\84\C3\8CB\80\07yx\07s\98q\0C\E6\00\0F\ED\10\0E\F4\80\0E3\0CB\1E\C2\C1\1D\CE\A1\1Cf0\05=\88C8\84\83\1B\CC\03=\C8C=\8C\03=\CCx\8Ctp\07{\08\07yH\87pp\07zp\03vx\87p \87\19\CC\11\0E\EC\90\0E\E10\0Fn0\0F\E3\F0\0E\F0P\0E3\10\C4\1D\DE!\1C\D8!\1D\C2a\1Ef0\89;\BC\83;\D0C9\B4\03<\BC\83<\84\03;\CC\F0\14v`\07{h\077h\87rh\077\80\87p\90\87p`\07v(\07v\F8\05vx\87w\80\87_\08\87q\18\87r\98\87y\98\81,\EE\F0\0E\EE\E0\0E\F5\C0\0E\EC0\03b\C8\A1\1C\E4\A1\1C\CC\A1\1C\E4\A1\1C\DCa\1C\CA!\1C\C4\81\1D\CAa\06\D6\90C9\C8C9\98C9\C8C9\B8\C38\94C8\88\03;\94\C3/\BC\83<\FC\82;\D4\03;\B0\C3\0C\C7i\87pX\87rp\83th\07x`\87t\18\87t\A0\87\19\CES\0F\EE\00\0F\F2P\0E\E4\90\0E\E3@\0F\E1 \0E\ECP\0E3 (\1D\DC\C1\1E\C2A\1E\D2!\1C\DC\81\1E\DC\E0\1C\E4\E1\1D\EA\01\1Ef\18Q8\B0C:\9C\83;\CCP$v`\07{h\077`\87wx\07x\98QL\F4\90\0F\F0P\0E3\1Ej\1E\CAa\1C\E8!\1D\DE\C1\1D~\01\1E\E4\A1\1C\CC!\1D\F0a\06T\85\838\CC\C3;\B0C=\D0C9\FC\C2<\E4C;\88\C3;\B0\C3\8C\C5\0A\87y\98\87w\18\87t\08\07z(\07r\98\81\\\E3\10\0E\EC\C0\0E\E5P\0E\F30#\C1\D2A\1E\E4\E1\17\D8\E1\1D\DE\01\1EfH\19;\B0\83=\B4\83\1B\84\C38\8CC9\CC\C3<\B8\C19\C8\C3;\D4\03<\CCH\B4q\08\07v`\07q\08\87qX\87\19\DB\C6\0E\EC`\0F\ED\E0\06\F0 \0F\E50\0F\E5 \0F\F6P\0En\10\0E\E30\0E\E50\0F\F3\E0\06\E9\E0\0E\E4P\0E\F80#\E2\ECa\1C\C2\81\1D\D8\E1\17\EC!\1D\E6!\1D\C4!\1D\D8!\1D\E8!\1Ff \9D;\BCC=\B8\039\94\839\CCX\BCpp\07wx\07z\08\07zH\87wp\87\19\CB\E7\0E\EF0\0F\E1\E0\0E\E9@\0F\E9\A0\0F\E50\C3\01\03s\A8\07w\18\87_\98\87pp\87t\A0\87t\D0\87r\98\81\84A9\E0\C38\B0C=\90C9\CC@\C4\A0\1D\CA\A1\1D\E0A\1E\DE\C1\1Cf$c0\0E\E1\C0\0E\EC0\0F\E9@\0F\E50C!\83u\18\07sH\87_\A0\87|\80\87r\98\B1\94\01<\8C\C3<\94\C38\D0C:\BC\83;\CC\C3\8C\C5\0CH!\15Ba\1E\E6!\1D\CE\C1\1DR\81\14fLg0\0E\EF \0F\EF\E0\06\EFP\0F\F40\0F\E9@\0E\E5\E0\06\E6 \0F\E1\D0\0E\E50\A3@\83vh\07y\08\87\19R\1A\B8\C3;\84\03;\A4C8\CC\83\1B\84\039\90\83<\CC\03<\84\C38\94\03\00\00\00\00y \00\00\19\00\00\00\92\1EH C\88\0C\19\09dd\C8\C9 \81\8C\042FFF\13\81B\A0\90\F1\C4\C8\089B\86\8Cb@,\00\00\00\07\00\00\00<stdin>\00#\08\010C \CC\10\042\12\98\A0\DC\D6\D2\E8\E6\EA\DC\CA\\\C8\CA\EC\D2\C6\CA\\\DA\DE\C8\EA\D8\CA\\\CC\D8\C2\CE\E6F\11\84\01\00\00\00\A9\18\00\00-\00\00\00\0B\0Ar(\87w\80\07zXp\98C=\B8\C38\B0C9\D0\C3\82\E6\1C\C6\A1\0D\E8A\1E\C2\C1\1D\E6!\1D\E8!\1D\DE\C1\1D\164\E3`\0E\E7P\0F\E1 \0F\E4@\0F\E1 \0F\E7P\0E\F4\B0\80\81\07y(\87p`\07vx\87q\08\07z(\07rXp\9C\C38\B4\01;\A4\83=\94\C3\02k\1C\D8!\1C\DC\E1\1C\DC \1C\E4a\1C\DC \1C\E8\81\1E\C2a\1C\D0\A1\1C\C8a\1C\C2\81\1D\D8a\C1\01\0F\F4 \0F\E1P\0F\F4\80\0E\0B\88u\18\07sH\87\05\CF8\BC\83;\D8C9\C8\C39\94\83;\8CC9\8C\03=\C8\03;\00\00\00\00\D1\10\00\00\06\00\00\00\07\CC<\A4\83;\9C\03;\94\03=\A0\83<\94C8\90\C3\01\00\00\00q \00\00\02\00\00\002\0E\10\22\04\00\00\00\00\00\00\00]\0C\00\00\11\00\00\00\12\03\94v\00\00\00\0021.1.3 a9a836d75454d0e801a53954ccbbe3a07732d70e<stdin>\00\00\00\00\00\00" #1

attributes #0 = { kit_bc kit_tt(2) }
attributes #1 = { kit_bc kit_tt(4) }
```

### Embedded Bitcode Tools

Kitsune provides tools that operate on embedded modules. More details about
these tools and their usage can be found by following the links in the table
below.

```{table}
| Tool | Summary |
| :--: | :------ |
| [kit-enc](CommandGuide/kit-enc.md) | Embedded bitcode into an empty host module |
| [kit-mbc](CommandGuide/kit-mbc.md) | Extract embedded bitcode from a host module |
```

### Embedded Bitcode Passes

[Embedded bitcode passes](glossary-embedded-bitcode-pass) operate on these
embedded bitcode modules. These passes can be added to LLVM's standard pass
pipeline that uses the [new pass manager](glossary-new-pass-manager). From
the perspective of this pass manager, these are
[module passes](glossary-module-pass) since they operate on a host module.
However, the only changes that they may make are to the initializers of global
variables in the module that contain embedded bitcode.

The implementation of these passes is opaque to pass developers. From their
perspective, these are still "module passes", though in this case, it is
because the unit on which they operate is a single embedded bitcode module.
While we do not do so, one could refer to these as "embedded module passes".

```{note}
There are currently no plans to provide "embedded function passes" that would
be analogous to [function passes](glossary-function-pass).
```

[This document](WritingEmbeddedBitcodePass.md) provides more information
about writing an embedded bitcode pass.


## Related Documentation

- [Pass Pipeline](PassPipeline.md)
- [Writing an Embedded Bitcode Pass](WritingEmbeddedBitcodePass)
- [Writing a Tapir Target Plugin](WritingTapirTargetPlugin)
