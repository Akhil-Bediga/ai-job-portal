"""A small TF-IDF vectorizer built on NumPy.

I implemented this by hand rather than pulling in scikit-learn: it's the core of
the recommender, the math is short, and it keeps the dependency list lean. The
weighting is the standard smoothed IDF, tf * idf, with L2-normalized rows so a
cosine similarity is just a dot product.
"""
from __future__ import annotations

import numpy as np


class TfidfVectorizer:
    def __init__(self, min_df: int = 1, max_df: float = 1.0) -> None:
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray = np.empty(0)

    def fit(self, docs: list[list[str]]) -> "TfidfVectorizer":
        n_docs = len(docs)
        df: dict[str, int] = {}
        for tokens in docs:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        max_count = self.max_df * n_docs if isinstance(self.max_df, float) else self.max_df
        kept = sorted(t for t, c in df.items() if self.min_df <= c <= max_count)

        self.vocabulary_ = {term: i for i, term in enumerate(kept)}
        idf = np.empty(len(kept), dtype=np.float64)
        for term, i in self.vocabulary_.items():
            idf[i] = np.log((1 + n_docs) / (1 + df[term])) + 1.0
        self.idf_ = idf
        return self

    def transform(self, docs: list[list[str]]) -> np.ndarray:
        vocab = self.vocabulary_
        mat = np.zeros((len(docs), len(vocab)), dtype=np.float64)
        for row, tokens in enumerate(docs):
            for term in tokens:
                j = vocab.get(term)
                if j is not None:
                    mat[row, j] += 1.0
        mat *= self.idf_                       # tf * idf, broadcast over columns
        return l2_normalize(mat)

    def fit_transform(self, docs: list[list[str]]) -> np.ndarray:
        return self.fit(docs).transform(docs)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0                    # leave all-zero rows as zeros
    return mat / norms


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of `a` and every row of `b`."""
    return l2_normalize(np.atleast_2d(a)) @ l2_normalize(np.atleast_2d(b)).T
