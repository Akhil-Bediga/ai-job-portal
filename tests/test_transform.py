"""Unit tests for the transform layer."""
import pandas as pd

from src.etl.extract import RawData
from src.etl.transform import _clean_jobs, transform


def _job_row(**overrides) -> dict:
    row = {
        "job_id": "j1", "title": "Data Engineer", "company": "Acme",
        "family": "Data Engineer", "city": "Austin", "state": "TX", "remote": 1,
        "employment_type": "FULL_TIME", "seniority": "SENIOR",
        "salary_min": 120000, "salary_max": 160000, "posted_date": "2026-05-01",
    }
    row.update(overrides)
    return row


def test_clean_jobs_dedupes_by_id():
    df = pd.DataFrame([_job_row(), _job_row(), _job_row(job_id="j2")])
    out = _clean_jobs(df)
    assert len(out) == 2


def test_salary_avg_is_midpoint():
    out = _clean_jobs(pd.DataFrame([_job_row(salary_min=100000, salary_max=200000)]))
    assert out.iloc[0]["salary_avg"] == 150000


def test_reversed_salary_bounds_are_swapped():
    out = _clean_jobs(pd.DataFrame([_job_row(salary_min=200000, salary_max=100000)]))
    assert out.iloc[0]["salary_min"] == 100000
    assert out.iloc[0]["salary_max"] == 200000


def test_invalid_enums_fall_back_to_defaults():
    out = _clean_jobs(pd.DataFrame([_job_row(employment_type="weird", seniority="boss")]))
    assert out.iloc[0]["employment_type"] == "FULL_TIME"
    assert out.iloc[0]["seniority"] == "MID"


def test_remote_is_clipped_to_binary():
    out = _clean_jobs(pd.DataFrame([_job_row(remote=5)]))
    assert out.iloc[0]["remote"] == 1


def test_transform_extracts_skills_and_builds_docs():
    raw = RawData(
        jobs=pd.DataFrame([_job_row(job_id="j1"), _job_row(job_id="j2")]),
        texts={
            "j1": "We need python and apache spark for our etl pipelines.",
            "j2": "Frontend role using react, typescript and css.",
        },
        resumes=[{"resume_id": "r1", "family": "Data Engineer",
                  "text": "python, sql, apache spark"}],
    )
    td = transform(raw)
    assert len(td.job_docs) == 2
    assert not td.job_skills.empty
    assert "apache spark" in set(td.job_skills["skill"])
    assert td.resumes[0]["skills"]  # resume skills extracted too
