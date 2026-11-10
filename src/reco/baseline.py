"""Naive keyword-overlap recommender - the thing we measure the real engine against.

It does the obvious first thing anyone tries: bag-of-words term-frequency cosine
on the raw text. No IDF, no stopword removal, no skill extraction. That means the
boilerplate every posting shares ("we are looking for a talented...") counts just
as much as the skills that actually distinguish roles, which is exactly why it
underperforms. See benchmarks/eval_reco.py for the numbers.
"""
from __future__ import annotations

import numpy as np

from src.nlp.preprocess import words
from src.nlp.vectorizer import l2_normalize
from src.reco.engine import JobDoc


class KeywordBaseline:
    def __init__(self) -> None:
        self.job_ids: list[str] = []
        self.vocabulary_: dict[str, int] = {}
        self.job_matrix: np.ndarray = np.empty(0)

    def _counts(self, docs: list[list[str]]) -> np.ndarray:
        mat = np.zeros((len(docs), len(self.vocabulary_)), dtype=np.float64)
        for row, toks in enumerate(docs):
            for tok in toks:
                j = self.vocabulary_.get(tok)
                if j is not None:
                    mat[row, j] += 1.0
        return l2_normalize(mat)

    def fit(self, jobs: list[JobDoc]) -> "KeywordBaseline":
        self.job_ids = [j.job_id for j in jobs]
        tokenized = [words(j.text) for j in jobs]            # raw tokens, stopwords kept
        vocab = sorted({t for toks in tokenized for t in toks})
        self.vocabulary_ = {t: i for i, t in enumerate(vocab)}
        self.job_matrix = self._counts(tokenized)
        return self

    def rank_all(self, resume_text: str) -> list[tuple[str, float]]:
        vec = self._counts([words(resume_text)])
        sim = (vec @ self.job_matrix.T).ravel()
        order = np.argsort(sim)[::-1]
        return [(self.job_ids[i], float(sim[i])) for i in order]

    def recommend(self, resume_text: str, top_n: int = 10) -> list[tuple[str, float]]:
        return self.rank_all(resume_text)[:top_n]
