"""Glue between the stores and the recommender for the API process.

Builds the recommender once (lazily, on first request) from the MongoDB
documents, and keeps a job_id -> structured-metadata map from MySQL so matches
can be returned with titles, companies, and salaries attached. Rebuilding is
cheap for this dataset; reload() is exposed for after a fresh pipeline run.
"""
from __future__ import annotations

from sqlalchemy import text

from src.db.mongo import get_db
from src.db.mysql import get_engine
from src.reco.engine import JobDoc, Recommender


class RecoService:
    def __init__(self) -> None:
        self._reco: Recommender | None = None
        self._meta: dict[str, dict] = {}

    def _build(self) -> None:
        docs = list(get_db().job_text.find({}, {"_id": 0, "job_id": 1, "text": 1, "skills": 1}))
        jobs = [JobDoc(d["job_id"], d.get("text", ""), d.get("skills", [])) for d in docs]
        self._reco = Recommender().fit(jobs)

        with get_engine().connect() as conn:
            rows = conn.execute(text("SELECT * FROM jobs")).fetchall()
        self._meta = {r._mapping["job_id"]: dict(r._mapping) for r in rows}

    def _ensure(self) -> Recommender:
        if self._reco is None:
            self._build()
        assert self._reco is not None
        return self._reco

    def reload(self) -> int:
        self._build()
        return len(self._meta)

    def metadata(self, job_id: str) -> dict:
        self._ensure()
        return self._meta.get(job_id, {})

    def is_ready(self) -> bool:
        return self._reco is not None

    def recommend(self, resume_text: str, top_n: int = 10) -> list[dict]:
        reco = self._ensure()
        out = []
        for m in reco.recommend(resume_text, top_n=top_n):
            meta = self._meta.get(m.job_id, {})
            out.append({
                "job_id": m.job_id,
                "title": meta.get("title"),
                "company": meta.get("company"),
                "family": meta.get("family"),
                "city": meta.get("city"),
                "state": meta.get("state"),
                "remote": bool(meta.get("remote", 0)),
                "seniority": meta.get("seniority"),
                "salary_avg": meta.get("salary_avg"),
                "score": m.score,
                "text_score": m.text_score,
                "skill_score": m.skill_score,
                "matched_skills": m.matched_skills,
                "missing_skills": m.missing_skills,
            })
        return out


service = RecoService()
