"""Load transformed data into both stores.

MySQL gets the structured, queryable columns (the API filters and aggregates on
these). MongoDB gets the unstructured side - full descriptions, extracted
skills, and resumes - which is what the recommender vectorizes at serve time.
Both loads are idempotent: truncate / replace, then insert.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from src.db.mongo import get_db
from src.db.mysql import get_engine
from src.etl.transform import TransformedData


def _load_mysql(td: TransformedData) -> dict[str, int]:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        conn.execute(text("TRUNCATE TABLE job_skills"))
        conn.execute(text("TRUNCATE TABLE jobs"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

    td.jobs.to_sql("jobs", engine, if_exists="append", index=False,
                   chunksize=1000, method="multi")
    td.job_skills.to_sql("job_skills", engine, if_exists="append", index=False,
                         chunksize=1000, method="multi")
    return {"jobs": len(td.jobs), "job_skills": len(td.job_skills)}


def _load_mongo(td: TransformedData) -> dict[str, int]:
    db = get_db()

    db.job_text.delete_many({})
    if td.job_docs:
        db.job_text.insert_many(td.job_docs)
    db.job_text.create_index("job_id", unique=True)
    db.job_text.create_index("skills")

    db.resumes.delete_many({})
    if td.resumes:
        db.resumes.insert_many(td.resumes)
    db.resumes.create_index("resume_id", unique=True)
    return {"job_text": len(td.job_docs), "resumes": len(td.resumes)}


def load(td: TransformedData) -> dict[str, int]:
    print("      loading MySQL (jobs, job_skills)")
    counts = _load_mysql(td)
    print("      loading MongoDB (job_text, resumes)")
    counts.update(_load_mongo(td))
    return counts
