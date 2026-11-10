"""Text cleaning and tokenization.

Everything here is dependency-free on purpose - no nltk corpora to download, no
spaCy model to ship. A bundled English stopword list plus a regex tokenizer is
enough for resume/job text, and it keeps the project runnable offline.
"""
from __future__ import annotations

import re

# keep letters, digits and the punctuation that shows up inside real skill names
# (c++, c#, node.js, a/b testing, ci-cd)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#./-]*")

STOPWORDS: frozenset[str] = frozenset("""
a an and are as at be been being but by for from had has have he her his if in
into is it its of on or our our s so such that the their them then there these
they this to was we were what when where which who will with you your yours about
above after again against all am any because before below between both did do does
doing down during each few further here how i me more most no nor not now off once
only other out over own same should some than too under until up very via while
""".split())


def words(text: str) -> list[str]:
    """Lowercase tokens, no stopword filtering. Skill matching needs every token.

    Trailing/leading . / - are stripped (sentence punctuation) but + and # are
    kept so "c++" and "c#" survive as tokens.
    """
    if not text:
        return []
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tok = tok.strip("./-")
        if tok:
            out.append(tok)
    return out


def tokenize(text: str) -> list[str]:
    """Tokens for the vectorizer: stopwords dropped, 1-char and pure-number tokens out."""
    out = []
    for tok in words(text):
        if tok in STOPWORDS or len(tok) < 2 or tok.isdigit():
            continue
        out.append(tok)
    return out
