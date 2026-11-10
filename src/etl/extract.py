"""Read the raw generated files off disk into memory."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from src.config import pipeline_cfg

RAW_DIR = pipeline_cfg.data_dir / "raw"


@dataclass
class RawData:
    jobs: pd.DataFrame          # structured posting fields
    texts: dict[str, str]       # job_id -> full description
    resumes: list[dict]         # [{resume_id, family, text}]


def extract() -> RawData:
    print(f"      reading raw files from {RAW_DIR}")
    jobs = pd.read_csv(RAW_DIR / "jobs.csv")
    texts = json.loads((RAW_DIR / "jobs_text.json").read_text())
    resumes = json.loads((RAW_DIR / "resumes.json").read_text())
    return RawData(jobs=jobs, texts=texts, resumes=resumes)
