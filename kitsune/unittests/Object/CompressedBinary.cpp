//===- CompressedBinary.cpp - Compressed binary objects for unit tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For the object-related unit tests, we compress a binary, then encode it in
// base64 so the raw object data can be included in the source files. This
// provides utilities to decompress such objects.
//
//===----------------------------------------------------------------------===//

#include "CompressedBinary.h"

#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

CompressedBinary::CompressedBinary(StringRef name, StringRef inBuf,
                                   uint64_t outSize)
    : inBuf(inBuf), outSize(outSize) {
  outBuf.resize_for_overwrite(outSize);

  DebugCompressionType zlib = DebugCompressionType::Zlib;
  std::vector<char> decoded;
  if (Error err = decodeBase64(inBuf, decoded))
    ADD_FAILURE();
  ArrayRef<uint8_t> in(reinterpret_cast<const uint8_t *>(decoded.data()),
                       decoded.size());
  uint8_t *out = reinterpret_cast<uint8_t *>(outBuf.data());
  if (Error err = compression::decompress(zlib, in, out, outSize))
    ADD_FAILURE();

  memBuf = MemoryBufferRef(StringRef(outBuf.data(), outSize), name);
}

template <typename T>
static std::unique_ptr<T> getIfOk(Expected<std::unique_ptr<T>> tOrErr) {
  if (not tOrErr) {
    fprintf(stderr, "Unexpected error creating global object");
    exit(1);
  }
  return std::move(*tOrErr);
}

static std::unique_ptr<Archive> a(const CompressedBinary &c) {
  return getIfOk(Archive::create(c.memBuf));
}

static std::unique_ptr<ObjectFile> o(const CompressedBinary &c) {
  return getIfOk(ObjectFile::createObjectFile(c.memBuf));
}

static const CompressedBinary cArEmpty("arEmpty",
                                       "eNpTtEksSs6w4wIACn4CRA==", 8);
const std::unique_ptr<Archive> arEmpty = a(cArEmpty);

static const CompressedBinary cArNoDeviceCode(
    "arNoDeviceCode",
    "eNpTtEksSs6w49JXQAUGGGwDFHELuGwCFwMUpOYWlFTq5esTMsPMxARMGxlYwM2od/VxY2JkhB"
    "nFwMhgx4DgIUAAEtsBSjKBVOoVlxSVJCYxUAmA7GbGIu4ApTmxqEcGAMBPGhs=",
    344);
const std::unique_ptr<Archive> arNoDeviceCode = a(cArNoDeviceCode);

static const CompressedBinary cArCuda1(
    "arCuda1",
    "eNpTtEksSs6w49JXQAUGGGwDFHELuGwCFwMUFOfGmxvp5esTMsPMxARMGxoYGMDMqHf1cWNiZI"
    "QZxcDIYMeA4CGABjOC7QAlmYEqeRiYwDyoOcbscHP2MTRgMccDargHqwfQBAuwOWwgG/WKM4pL"
    "ikoSk4AsGF2Zi0THF2fkpVQw6OWV6WXmpeUz6KWkJpWmx6cVJeamgkWTE3Ny0osSCzLAvIKi/J"
    "L8ksqCVJoajRPYAzEzAytWuf9AAKL/Qem/UPoPlKYEMILtxQQOUDoai3pkwI1D/2wi9QsDMRMW"
    "/T+gtAGUZoJiDihfAkobYTETnP4YcfsXPdyBYgW49CtA+TA/skBpmDsQSSQ7s0QvOT8lVS+5NC"
    "WRgVbxIsCM6n9c/uKEijHhiNcLaBIcaOoAPJ54rw==",
    1136);
const std::unique_ptr<Archive> arCuda1 = a(cArCuda1);

static const CompressedBinary cArHip1(
    "arHip1",
    "eNpTtEksSs6w49JXQAUGGGwDFHELuGwCFwMUpKdVWBqY6eXrEzDDzMQETBsamJnAzKh39XFjYm"
    "SEGcXAyGDHgOAhQAYzgu0AJZmBKoWgqqHmODDDzXmA1RwOqKA+K8IcDqBKdiDLG4gVgNjR18U9"
    "IJSBoXldYm5KRnGiXnZqUV5qTvGEtVB+SWJRemrJTiAvPTlPF0jpQiR0dSFBAdNXllpUnJmfN4"
    "mRiQEn0CtJrShh0IPYqZdeUBSfm1iRmVuaW8ygl5dfkqrn7heqW1ySmJzNoJeTU5Ybn5iSAjQ1"
    "HSLLoFdcUlSSmASkK3NBNKUAFCbMWMR3QOkANHFGLHwQZsNiLjbAgsZnx2ImPv3oamUo1G8FdQ"
    "MTDv0JBNyvAwo/n//5QKoBW/gh+xOb/R5Y7AaBBVBaAi2cOdDEGfSyM0v0kvNTUvUyMgvgyYNi"
    "wI8jXQRABcWJTBe4wlUAzXAONHUArFVkYA==",
    1200);
const std::unique_ptr<Archive> arHip1 = a(cArHip1);

static const CompressedBinary cArMulti(
    "arMulti",
    "eNq1VE9v0zAUd5J1KwOJaRzgwGHcOBA3jKoMhEaQgFVioF524pB5iZtEzT/FbtWdmMQXQFy58A"
    "XgDBc+Ax+Bj8EBGG760rRpTIuqPcn5+T37/Rzbz79bj0hqe/ubjZ1ZM+b6xkx8bzJ6vInAWGjd"
    "38VxYxFHq9nM8K5hGDnHm2eHz1VFyamQgvZR4RV2Wyv6Jnw1MfMKUjMPeO5tTHi+obMKnjaQt2"
    "ttwbCX8ayPVsTMYzzl5ET0cjwNp9BiXuQMEY4G2I+6McIOPem7VjclIc2iNgkCNyWJl3lJGvOY"
    "nyb0Qqml9lg0DdUqx86FjfAP4G/AX4CrmJKtO28m4OuK+dN2WZL/Ycn8a6KpFfk/AQ1AFVod/B"
    "uAuxWcWf0p8v2Wz13EEln+Dvj5HtcA8/8oSqTnc2zHDsV23yHoou5lS5vdv2xflyCmSu71e2mg"
    "XprndocPjFYhEos1otX8X43wJBqxDbOBx9QmPD8qeeoQbNQKnrqYOdKWF3CHT14+PegcIfT2Mw"
    "kdjxHco2lEA/buE/icpC7lX4Xn2pEuQB8P6Pr4KPK8AU2ZH0fvFVV+eZjTIUd4vCZ2k9QKydAP"
    "+yETehBzig9eHemME7uHcBAMQos4jmB1x6NzqrOqmZJa+gLYWVBLCrR1SS2Vba3kb0jeqImWe6"
    "M3V8x/CP8gewvHC/7/zuj8Ds9jAWdV5ze9z6r12xKN+1jSsvycyxo3pS2en0zKY2W7KqmLDgSv"
    "L1kXsnPd0v6tMX8BseTSug==",
    2260);
const std::unique_ptr<Archive> arMulti = a(cArMulti);

static const CompressedBinary cArHetero(
    "arHetero",
    "eNpTtEksSs6w49JXQAUGGGwDFHELuGwCFwMUpOYWlFTq5esTMsPMxARMGxlYwM2od/VxY2JkhB"
    "nFwMhgx4DgIUAAEtsBSjKBVOoVlxSVJCYxUAmA7GbGIu4ApTmxqEcGkLAoqSjRJy4sDJHCk4sL"
    "AK9BJQQ=",
    406);
const std::unique_ptr<Archive> arHetero = a(cArHetero);

static const CompressedBinary elfExecDynamicC(
    "elfExecDynamic",
    "eNrtmt9LVEEUx+fuD11N260UTCUlRAraXdZfGOEPUvNKGmb64EPeVl1tYV1DVxJ7SLKgRYQC6b"
    "k/IcKXnlpRRN/0rUcNBQOtjQqswNvM3jO7cycXpBcJzkfW7z1nzjkzd+a6eJl53NJ+w6IohGMl"
    "9YRZpS7DbiR/00hqyWm4zqAfmxAnq8tqVkeyHyNvHvyyFhOzKoLaSHp27WYlrlSeXbBl9SlmFf"
    "MS/ZWCX9LnxKxiHpsblwfc9WYlFpgPiznPAnmlkFdab9bXiln5fNrg0wnzJ6s8fDnvAcTJWkbM"
    "apPy7kNBWeX+eN6dncjgv4yzE/Js0CCr/Lzw/m7TvAxyfPhj0QX9pVu/uGJWRejXBc9a660eYp"
    "kmMZvUngfPP2t/P9+7sPdQySy6ub21PLyf0bM20usNBftrqryhQXcoGJ6YdE/W1rhrqjzjo56K"
    "RB1FqHdcWPzZI/xNwlyLaGn84TT+c/Rz8aiONY3ezYA2HvGPRbQRfzBMPQOTfm0oGPaHglMBwp"
    "rZvdWQ1va2601ahafCU528rqwiWlt3hzYYGAsMB8cjgbHujqbQaDjQ7e8PBWit4ZHRMFTXjNAj"
    "A9lflpL4scJ8WBLj5es9URjMYqtyBXwLL19lsJxaYl53/jy6zhhaIPljUNDRaPZzO9aQ+t4U13"
    "BV8FsE/7rgtwr+Dw3m71/OpuC3EwRBEARBEAQx88154Zc6s+dQZ+2LXkLUZ7GIRV9XZ5YdS8l2"
    "335bdKNPjX5UZ7bjnd0tvphvTZ2r26LNev4jmvV9yOMseZqIp/53rM6c/S2TqweRfFq6D0pn6Z"
    "vOkmkWtwRK46OJ+OonTC4fqtG4uvi5QV08sKrKirpxGMmjBdxQwKFvDjlLmlP5bHzTdS1e9i98"
    "ZY86U5fNLtXoTiRXna0r97IhHtB3uN0BXddX7MXUodxdSo2X1mkkQymb1fs0TGOhvi+2lJqj5K"
    "wgCIIgCIIgCIIgyP8F2wdTiqzXCoix71Qc1/Uqpl91vZ0Ye5WJuKkuoky6lKKcTMcLxfCzvdbN"
    "L7pezgKaMxM7X+wXy1+ldcT9Kr4vyva82L4n99lwCRAEQRAEQRAEQU4Efq4xJpxjZvCznTnJQE"
    "NywSyDvPO8Gc5FFoHNz0EWgvJzk8VS+49DfZTpG3h55GcY78GLIj/Luw7t2WDz89eneH3QfOn+"
    "MkEv2c39xkCzpPzfujEeHnoI9gGMRwebz1Mc7F1o/wm29YTXlZ8fTwc/n/oHZVAGWQ==",
    12304);
const std::unique_ptr<ObjectFile> elfExecDynamic = o(elfExecDynamicC);

static const CompressedBinary elfExecStaticC(
    "elfExecStatic",
    "eNrtzTEOgCAQRNGFSEMjV7GzMIZGY+GdvJKlt1NEiAQr+/8KJgO7YZvWWSslmZZRYnM+di9fXv"
    "ow97hnm+q1zD21nOLePVP0/F9Om65ttQcAAAAAAP47hnAs3dmaCzNeB/U=",
    4106);
const std::unique_ptr<ObjectFile> elfExecStatic = o(elfExecStaticC);

static const CompressedBinary elfSharedC(
    "elfShared",
    "eNrtms9rE0EUx2fzy41aUqQFrdIU8VChpBCpRCFpsL+m0BYsiXgpS5omYTFNIFlBvBiMCiVUiv"
    "g3iHjz5s2GSOlJol6Kp3goKBQpiND2kHUn+8bMbtOiJw++T9h85715b37t7mHhPZiYmXRIEuE4"
    "SYS0LUKi5DBREiIytFmsixyDy6bd7Ty3YNt1n1hVzGsNNQB+m9qnE/M8xhUKmHYoYtU6bHpLsu"
    "Y5II9CHo1YtSRZVRbmd/EBOuglYlW+3pvb2pKH/Dn82OaNPNcx+9uUrCoJ8/a07jshU3Nx8uRM"
    "Xn3Wf+vTxIvn85WP78mb4MO3PF4S8v5mfQMd/NIR/qEj/JPGdbHTBIqSWc7nlKKWKGiKQpTp2K"
    "yylCqkMmpRSxVis2PZfC4VSyxmU2Zf5x4leS+hpNVcIqveT5GsupgMFPOBq2RqZvrGmBIMBAMj"
    "5qrZzwGtuHD+d/tUL7u5CrGeO38eGhAYtfnXuq0vmmx78TZHTfXYzr4u+B2Cf0vwOwV/Q/C7CI"
    "IgCIIgCIIg/yM/fP0HtLwj04q7Omx81j5e1xx6nZbfyTXzO3c1/IX5V92fmVzb03qN0KcQ6tUb"
    "Pn+JxdVAjfjXrfiRV0wuN+nKLq1+H6XVPSeVNuiHptZjDLAAA8h6I+3zj7fz2XpK4eVh9kl1JU"
    "7L4eusSVe2tS5aCc8Zht47aPx/Teq6vuEeN5rSQi0d8PkfEVhHlKTbNhvvW8aI9fnbe/29OwRB"
    "EARBEARBEARBEARBEARBEARBEITXs64J9dWMOOhpHgj9XWDugH0ObF4nex5sXv/aB8rrZS/Y+n"
    "829TzTQSh+dYN/HWxeQ03BPgn2HdBToGdBT4DedlrneQnqtcU3dXP+EMTrYPNz2AV7CPr3wXb+"
    "o/vF69UPETXlF/CkuGk=",
    12296);
const std::unique_ptr<ObjectFile> elfShared = o(elfSharedC);

static const CompressedBinary
    elfEmptyC("elfEmpty", "eNqrd/VxY2JkZIABRgY7BgQPO3BAYgMAY34B2w==", 64);
const std::unique_ptr<ObjectFile> elfEmpty = o(elfEmptyC);

static const CompressedBinary elfCudaC(
    "elfCuda",
    "eNqrd/VxY2JkZIABRgY7BgQPATSYEWwHKMkMVMnDwATm1UPMMWaHm7OPoQGLOR5Qwz1YPYAmWI"
    "DNYQPZqFecUVxSVJKYBGTB6MpcJDq+OCMvpYJBL69MLzMvLZ9BLyU1qTQ9Pq0oMTcVLJqcmJOT"
    "XpRYkAHmFRTll+SXVBak0tRonMAeiJkZWLHK/QcCEP0PSv+F0n+gNCWAEWwvJnCA0tFY1CMDbh"
    "z6ZxOpXxiImbDo/wGlDaA0ExRzQPkSUNoIi5ng9MeI27/o4Q4UK8ClXwHKh/mRBUrD3IFIItmZ"
    "JXrJ+SmpesmlKYkMtIoXAWZU/+PyFydUjAlHvF5Ak+BAUwcADjlkjg==",
    1000);
const std::unique_ptr<ObjectFile> elfCuda = o(elfCudaC);

static const CompressedBinary elfHipC(
    "elfHip",
    "eNqrd/VxY2JkZIABRgY7BgQPATKYEWwHKMkMVCkEVV0PMceBGW7OA6zmcEAF9VkR5nAAVbIDWd"
    "5ArADEjr4u7gGhDAzN6xJzUzKKE/WyU4vyUnOKJ6yF8ksSi9JTS3YCeenJebpAShcioaubnlZh"
    "aWAG01eWWlScmZ83iZGJASfQK0mtKGHQg9ipl15QFJ+bWJGZW5pbzKCXl1+SqufuF6pbXJKYnM"
    "2gl5NTlhufmJICNDUdIsugV1xSVJKYBKQrc0E0pQAUJsxYxHdA6QA0cUYsfBBmw2IuNsCCxmfH"
    "YiY+/ehqZSjUbwV1AxMO/QkE3K8DCj+f//lAqgFb+CH7E5v9HljsBoEFUFoCLZw50MQZ9LIzS/"
    "SS81NS9TIyC+DJg2LAjyNdBEAFxYlMF7jCVQDNcA40dQDReVAZ",
    1064);
const std::unique_ptr<ObjectFile> elfHip = o(elfHipC);

static const CompressedBinary elfMultiC(
    "elfMulti",
    "eNq1U71O40AQXju/EhIgKKCgSIUo8IJAQkABRoID6QCloaIIi72xrfhP9iYKFZF4gdO1NPcCdz"
    "U0PMM9Ao9Bwd/aGcfBeBWkiE9af57dnZnd2W+uD45/yJKEEkhoB6VWiqNy+q/Ct8h3TiI5tq77"
    "cdYrgzgPqJcXB4LXS3UeYTOOU44y4tAMWcDIJf9L+MoZ4kZounoXYbeDLbfpIazTy7bRaAbEof"
    "GsRmzbCIhvxpYfeMxjVz791tBC7PJRQKXctTeOiF+BX4CfgceBFOf9DBX4PGf/MCYE/rdf9J/l"
    "Q87xfwJeBZZhVMGeB17LiRlhSRLfN1t3PueL/GtgJ3csAifnmIGIoGe1MMjzmHuuKkyulNK+qP"
    "KdUR/8hHx7J/uH9TOEbv4RRzdDgls0cKkd/voLNiOBQdk9twzNVTgp/QVFMZrdrdWNxK9Dg9Dy"
    "3N+SLBYAZrTLEO7nxIYfNBzStZy2E3Lteoziw9MzJWREayFs2x2nQXSdRzX6q586ZFyoAj3dAd"
    "dHvKcEoyzQcxbFjF0R6ElFX9PTwpj+23AGWeB/MeL8y1H9jt88Tr28+g3fMy//kaAf/2T6Lqlz"
    "th8RblkMa55OsWn5qTwGs1pbJ2h8TAl0UoOHXxxR5zmYE9X5f2ahKtBZ1n+68JFF/u/vHK8o",
    1864);
const std::unique_ptr<ObjectFile> elfMulti = o(elfMultiC);

static const CompressedBinary machOExecC(
    "machOExec",
    "eNrtmb9LQlEUx8/1qUmZPajBfgyviLBJXeKNRUpKUiIKEcTN8oWCGpiSNbkUNATtLfafBA2N/Q"
    "EtLY3R3lDd6zvizR7l0BTnAwc913e+75xzz+K9D28v70MATAMAlzC/sCfx5QwMkEwKSwjjPL2y"
    "Ft+OZzbhGwx+R+qYTOpk41tZh3ijLwB9D5qGy5zXrWa991i/nqHZej7Fd6u6bmipLueN6nGpWu"
    "Cl6sGhg14I9Uy957t+qJNzq8gPavmK5Zzf7ritt6z4mhI/Ipb6+2b3P5XcWI/Hktlefzp907/2"
    "a1fpqbovcwCtyAD7JHUiOAtBWTe+J2HYvsgP0g77P8gMOBHA1OXchRtHtXC5tBcunJQL+Ps05p"
    "FKxa5zuWjs5l67jDxOLc13S3cPdywk6gvifqss4HMRw56JRfRNY7D8comL5+jtzp2sbwbrNJSW"
    "Y5/c8r0m2BbCHEycW7+I0oEgCIIgCIIgCIIgiP/KBH6y0wywps50/5DvigHI8wt5ntB+/ejQPa"
    "+ClYB+7lod9VLrCIIgCIIgCIL4axgHD7gq+VIVZnmlyK2mtd+oW7xo5QtWDRY0aHuVG+82/jGR"
    "K2PMvtW070S5QzTAJ4BQWDE=",
    8288);
const std::unique_ptr<ObjectFile> machOExec = o(machOExecC);

static const CompressedBinary machOObjectC(
    "machOObject",
    "eNo7/+vtP3YGBkZmBiABxExAvAKIGRQYwEASiGcwEAYn0PjsUMwI5cfHl6RWlCDk4+NDXCNCiD"
    "cPCBqQOSpALABisHCB+QAQpQwj",
    200);
const std::unique_ptr<ObjectFile> machOObject = o(machOObjectC);

static const CompressedBinary machOSharedC(
    "machOShared",
    "eNo7/+vtP3YGBkZmBgYGNiDmAuIFjAwMrQwCDCAgCcQeQBwfH+IaEcJANGCFYhhAmOPj6eft6u"
    "IJM0sAlT7AiCrMCMUwoMTA0GBAvDMYmIBYAohPQA0B0RxAmhuIAxioB3iB2ABqFzLQL8kt0E/U"
    "y01MzsjXS6nMyUyCiEtD1fr4uJiFhhq6LIy5Ffj9ezaLCszvLFxgrAblH4C6WxPKP8GIao8ClA"
    "YArX8e8g==",
    488);
const std::unique_ptr<ObjectFile> machOShared = o(machOSharedC);
