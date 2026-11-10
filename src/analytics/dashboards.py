"""Generate Matplotlib dashboards from the warehouse.

Structured charts come straight from MySQL; the match-score chart runs the live
recommender over the resumes in MongoDB. PNGs land in docs/dashboards/ and are
embedded in the README.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.db.mongo import get_db
from src.db.mysql import get_engine
from src.reco.engine import JobDoc, Recommender

OUT_DIR = PROJECT_ROOT / "docs" / "dashboards"
ACCENT = "#2e6cf6"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
})


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_top_skills() -> Path:
    df = pd.read_sql("""
        SELECT skill, COUNT(*) AS demand
        FROM job_skills GROUP BY skill ORDER BY demand DESC LIMIT 15
    """, get_engine()).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df["skill"], df["demand"], color=ACCENT)
    ax.set_title("Top 15 In-Demand Skills", fontsize=14, fontweight="bold")
    ax.set_xlabel("postings requiring the skill")
    return _save(fig, "top_skills.png")


def chart_salary_by_family() -> Path:
    df = pd.read_sql("""
        SELECT family, ROUND(AVG(salary_avg)) AS avg_salary
        FROM jobs GROUP BY family ORDER BY avg_salary
    """, get_engine())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["family"], df["avg_salary"], color=ACCENT)
    ax.set_title("Average Salary by Role", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(lambda x, _: f"${x/1000:.0f}k")
    return _save(fig, "salary_by_family.png")


def chart_jobs_per_family() -> Path:
    df = pd.read_sql("""
        SELECT family, COUNT(*) AS postings
        FROM jobs GROUP BY family ORDER BY postings DESC
    """, get_engine())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["family"], df["postings"], color=ACCENT)
    ax.set_title("Open Postings by Role", fontsize=14, fontweight="bold")
    ax.set_ylabel("postings")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, "jobs_per_family.png")


def chart_salary_distribution() -> Path:
    df = pd.read_sql("SELECT salary_avg FROM jobs", get_engine())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["salary_avg"], bins=30, color=ACCENT, alpha=0.85)
    ax.set_title("Salary Distribution (all postings)", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(lambda x, _: f"${x/1000:.0f}k")
    ax.set_ylabel("postings")
    return _save(fig, "salary_distribution.png")


def chart_match_scores() -> Path:
    docs = list(get_db().job_text.find({}, {"_id": 0, "job_id": 1, "text": 1, "skills": 1}))
    reco = Recommender().fit([JobDoc(d["job_id"], d.get("text", ""), d.get("skills", [])) for d in docs])
    resumes = list(get_db().resumes.find({}, {"_id": 0, "text": 1}))
    top_scores = [reco.recommend(r["text"], top_n=1)[0].score for r in resumes]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(top_scores, bins=20, color=ACCENT, alpha=0.85)
    ax.axvline(np.mean(top_scores), color="#d9534f", linestyle="--",
               label=f"mean {np.mean(top_scores):.2f}")
    ax.set_title("Best-Match Score Distribution (per resume)", fontsize=14, fontweight="bold")
    ax.set_xlabel("recommender score")
    ax.set_ylabel("resumes")
    ax.legend()
    return _save(fig, "match_scores.png")


def main() -> None:
    charts = [
        chart_top_skills,
        chart_salary_by_family,
        chart_jobs_per_family,
        chart_salary_distribution,
        chart_match_scores,
    ]
    for fn in charts:
        path = fn()
        print(f"    wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
