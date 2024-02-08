# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pygments.lexer import RegexLexer, words
from pygments.lexers.c_cpp import CFamilyLexer

# Kitsune-specific keywords
kitsune_keywords = ('forall',)

# Add kitsune-specific keywords to the tokens in the C-family lexer.
def get_tokens():
    tokens = CFamilyLexer.tokens

    # The third element of the list associated with the 'keywords' key is a pair
    # of words i.e. keywords and a token type. w.words is a tuple of the
    # keywords.
    w, t = tokens['keywords'][2]

    keywords = tuple(sorted(w.words + kitsune_keywords))
    tokens['keywords'][2] = (words(keywords, w.suffix), t)

    return tokens

# This is a lexer for C and C++ that contains Kitsune-specific extensions. This
# approach is more than a little ugly, but it is quick.
class KitsuneCFamilyLexer(CFamilyLexer):
    name = 'KitsuneCpp'
    aliases = ['kitc', 'kitcpp']
    filenames = ['*.kit.c', '*.kit.cpp']

    tokens = get_tokens()

KitsuneCFamilyLexer()
