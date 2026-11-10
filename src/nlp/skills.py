"""Gazetteer-based skill extraction.

Pulls a known-skills list from data/skills.txt and finds them in free text.
Multi-word skills ("machine learning", "apache spark") are matched first and the
matched tokens are consumed, so "apache spark" doesn't also report a bare
"spark". Simple, transparent, and good enough - the messy real-world
alternative (NER on resumes) is overkill for a controlled dataset.
"""
from __future__ import annotations

from functools import lru_cache

from src.config import PROJECT_ROOT
from src.nlp.preprocess import words

_GAZETTEER_PATH = PROJECT_ROOT / "data" / "skills.txt"


@lru_cache(maxsize=1)
def load_skills() -> list[str]:
    skills = []
    for line in _GAZETTEER_PATH.read_text().splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            skills.append(line)
    # longest (in tokens, then chars) first so phrases win over their substrings
    skills.sort(key=lambda s: (len(s.split()), len(s)), reverse=True)
    return skills


def extract_skills(text: str) -> list[str]:
    toks = words(text)
    consumed = [False] * len(toks)
    found: set[str] = set()
    for skill in load_skills():
        parts = skill.split()
        n = len(parts)
        i = 0
        # consume every occurrence so a repeated "apache airflow" can't leave a
        # stray "airflow" token for the shorter gazetteer entry to match
        while i <= len(toks) - n:
            if not any(consumed[i:i + n]) and toks[i:i + n] == parts:
                found.add(skill)
                for j in range(i, i + n):
                    consumed[j] = True
                i += n
            else:
                i += 1
    return sorted(found)
