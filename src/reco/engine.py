"""The recommender.

Two signals are combined into a match score:

  text   - cosine similarity between the TF-IDF vector of the resume and each
           job description. Catches phrasing the gazetteer misses.
  skills - Jaccard overlap of extracted skills. Catches the hard requirements a
           resume either has or doesn't.

final = TEXT_WEIGHT * text + SKILL_WEIGHT * skills

Both parts run on the same tokenized text, so a single fit() builds everything.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.nlp.preprocess import tokenize
from src.nlp.skills import extract_skills
from src.nlp.vectorizer import TfidfVectorizer

TEXT_WEIGHT = 0.6
SKILL_WEIGHT = 0.4


@dataclass
class JobDoc:
    job_id: str
    text: str
    skills: list[str] = field(default_factory=list)


@dataclass
class Match:
    job_id: str
    score: float
    text_score: float
    skill_score: float
    matched_skills: list[str]
    missing_skills: list[str]


class Recommender:
    def __init__(self, text_weight: float = TEXT_WEIGHT, skill_weight: float = SKILL_WEIGHT) -> None:
        self.text_weight = text_weight
        self.skill_weight = skill_weight
        self.job_ids: list[str] = []
        self.job_skills: list[set[str]] = []
        self.vectorizer = TfidfVectorizer(min_df=2)
        self.job_matrix: np.ndarray = np.empty(0)

    def fit(self, jobs: list[JobDoc]) -> "Recommender":
        self.job_ids = [j.job_id for j in jobs]
        self.job_skills = [set(j.skills) for j in jobs]
        self.job_matrix = self.vectorizer.fit_transform([tokenize(j.text) for j in jobs])
        return self

    def _scores(self, resume_text: str) -> tuple[np.ndarray, np.ndarray, set[str]]:
        resume_vec = self.vectorizer.transform([tokenize(resume_text)])  # (1, V), normalized
        text_sim = (resume_vec @ self.job_matrix.T).ravel()             # cosine, since L2-normalized

        resume_skills = set(extract_skills(resume_text))
        skill_sim = np.array([_jaccard(resume_skills, js) for js in self.job_skills])
        return text_sim, skill_sim, resume_skills

    def rank_all(self, resume_text: str) -> list[tuple[str, float]]:
        text_sim, skill_sim, _ = self._scores(resume_text)
        combined = self.text_weight * text_sim + self.skill_weight * skill_sim
        order = np.argsort(combined)[::-1]
        return [(self.job_ids[i], float(combined[i])) for i in order]

    def recommend(self, resume_text: str, top_n: int = 10) -> list[Match]:
        text_sim, skill_sim, resume_skills = self._scores(resume_text)
        combined = self.text_weight * text_sim + self.skill_weight * skill_sim
        order = np.argsort(combined)[::-1][:top_n]

        matches = []
        for i in order:
            job_sk = self.job_skills[i]
            matches.append(Match(
                job_id=self.job_ids[i],
                score=round(float(combined[i]), 4),
                text_score=round(float(text_sim[i]), 4),
                skill_score=round(float(skill_sim[i]), 4),
                matched_skills=sorted(resume_skills & job_sk),
                missing_skills=sorted(job_sk - resume_skills),
            ))
        return matches


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
