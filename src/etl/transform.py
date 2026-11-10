"""Clean the postings and pull features out of the text.

Structured columns are coerced and validated for the MySQL side. The free text
is run through the skill gazetteer so each posting gets a clean skill list - that
list feeds both the job_skills table and the MongoDB documents the recommender
reads at serve time.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.etl.extract import RawData
from src.nlp.skills import extract_skills

JOB_COLUMNS = [
    "job_id", "title", "company", "family", "city", "state", "remote",
    "employment_type", "seniority", "salary_min", "salary_max", "salary_avg",
    "posted_date",
]
EMPLOYMENT_TYPES = {"FULL_TIME", "PART_TIME", "CONTRACT"}
SENIORITIES = {"JUNIOR", "MID", "SENIOR", "LEAD"}


@dataclass
class TransformedData:
    jobs: pd.DataFrame          # structured, MySQL-shaped
    job_skills: pd.DataFrame    # (job_id, skill), one row per pair
    job_docs: list[dict]        # MongoDB job_text documents
    resumes: list[dict]         # MongoDB resume documents


def _clean_jobs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["job_id"]).copy()

    df["title"] = df["title"].fillna("Untitled").str.slice(0, 160)
    df["company"] = df["company"].fillna("Unknown").str.slice(0, 120)
    df["family"] = df["family"].fillna("Other").str.slice(0, 60)
    df["city"] = df["city"].fillna("Unknown").str.slice(0, 80)
    df["state"] = df["state"].fillna("NA").str.slice(0, 40)
    df["remote"] = pd.to_numeric(df["remote"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    df["employment_type"] = df["employment_type"].fillna("FULL_TIME").str.upper()
    df.loc[~df["employment_type"].isin(EMPLOYMENT_TYPES), "employment_type"] = "FULL_TIME"
    df["seniority"] = df["seniority"].fillna("MID").str.upper()
    df.loc[~df["seniority"].isin(SENIORITIES), "seniority"] = "MID"

    df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce").fillna(0).astype(int)
    df["salary_max"] = pd.to_numeric(df["salary_max"], errors="coerce").fillna(0).astype(int)
    # don't trust the bounds blindly - swap if reversed, then derive the midpoint
    swap = df["salary_min"] > df["salary_max"]
    df.loc[swap, ["salary_min", "salary_max"]] = df.loc[swap, ["salary_max", "salary_min"]].to_numpy()
    df["salary_avg"] = ((df["salary_min"] + df["salary_max"]) / 2).round().astype(int)

    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce").dt.date
    return df[JOB_COLUMNS]


def _extract_features(jobs: pd.DataFrame, texts: dict[str, str]) -> tuple[pd.DataFrame, list[dict]]:
    skill_rows: list[dict] = []
    docs: list[dict] = []
    for row in jobs.itertuples(index=False):
        text = texts.get(row.job_id, "")
        skills = extract_skills(text)
        docs.append({
            "job_id": row.job_id,
            "family": row.family,
            "text": text,
            "skills": skills,
        })
        skill_rows.extend({"job_id": row.job_id, "skill": s} for s in skills)
    return pd.DataFrame(skill_rows, columns=["job_id", "skill"]), docs


def transform(raw: RawData) -> TransformedData:
    print("      cleaning postings")
    jobs = _clean_jobs(raw.jobs)

    print("      extracting skills from text")
    job_skills, job_docs = _extract_features(jobs, raw.texts)
    print(f"      {len(job_skills):,} job-skill pairs across {len(jobs):,} postings")

    resumes = [
        {"resume_id": r["resume_id"], "family": r["family"], "text": r["text"],
         "skills": extract_skills(r["text"])}
        for r in raw.resumes
    ]

    return TransformedData(jobs=jobs, job_skills=job_skills, job_docs=job_docs, resumes=resumes)
