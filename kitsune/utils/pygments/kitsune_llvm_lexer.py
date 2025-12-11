# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pygments.lexer import RegexLexer, words
from pygments.lexers.asm import LlvmLexer

# Kitsune-specific attributes and keywords
kitsune_attrs = ('kit_bc', 'kit_fb', 'kit_tt')

# Tapir-specific keywords.
tapir_keywords = ('detach', 'reattach', 'sync', 'within')

# Add kitsune-specific attributes and tapir-specific keywords to the tokens
# in LLVM's Lexer.
def get_tokens():
    tokens = LlvmLexer.tokens

    # The first element of the list associated with the 'keyword' key is a pair
    # of words i.e. keywords and a token type. w.words is a tuple of the
    # keywords.
    w, t = tokens['keyword'][0]

    keywords = tuple(sorted(w.words + kitsune_attrs + tapir_keywords))
    tokens['keyword'][0] = (words(keywords, w.suffix), t)

    return tokens

# This is a lexer for LLVM IR that contains Kitsune-specific extensions. This
# includes tapir instructions and kitsune-specific attributes and other
# keywords. To make it easier to keep this up-to-date with LLVM, we just use
# the LLVM-IR lexer and append kitsune-specific keywords to it. This is good
# enough for now.
class KitsuneLLVMLexer(LlvmLexer):
    name = 'KitsuneLLVM'
    aliases = ['kitll']
    filenames = ['*.kit.ll']

    tokens = get_tokens()
