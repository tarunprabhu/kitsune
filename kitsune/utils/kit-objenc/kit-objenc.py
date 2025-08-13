#!/usr/bin/env python3
#
#-------------------------------------------------------------------------------
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ------------------------------------------------------------------------------
#
# Utility to strip and compress an object file, then encode it as a C string.
# This is primarily used to generate input files for the object-related unit
# tests in kitsune/unittests/Object.
#
# ------------------------------------------------------------------------------

import argparse
import base64
import os
import tempfile
import sys
import zlib

def main() -> int:
    ap = argparse.ArgumentParser(
        prog = 'kit-objenc',
        description = 'Strip, compress and encode an object file',
        epilog = 'Compression uses zlib. Encoding is standard base64'
    )
    ap.add_argument(
        '--remove-sections',
        default = [],
        nargs = '+',
        help =
        'Strip the given sections. When using this command, use -- to separate '
        'the name of the last section to be removed and the name of the object '
        'file'
    )
    ap.add_argument(
        '--strip-sections',
        action = 'store_true',
        help = 'Strip sections from the object file'
    )
    ap.add_argument(
        'infile',
        metavar = '<object-file>',
        help = 'The object file to strip, compress and encode'
    )
    args = ap.parse_args()

    infile = args.infile
    bindir = os.path.dirname(sys.argv[0])
    llvm_strip = os.path.join(bindir, 'llvm-strip')

    if not os.path.exists(args.infile):
        print(f"No such file or directory: '{infile}'", file = stderr)
        sys.exit(1)

    temp = tempfile.NamedTemporaryFile(suffix = '.o')
    strip_args = [llvm_strip, '--strip-all', '-o', temp.name]
    if args.strip_sections:
        strip_args.append('--strip-sections')
    for section in args.remove_sections:
        strip_args.append(f'--remove-section={section}')
    strip_args.append(infile)

    os.system(' '.join(strip_args))

    obj = open(temp.name, 'rb').read()
    compressed = zlib.compress(obj, level = 9)
    s = base64.b64encode(compressed).decode('utf-8')

    print(f'encoded: {s}')
    print(f'decompressed size: {len(obj)}')

    return 0

if __name__ == '__main__':
    sys.exit(main())
