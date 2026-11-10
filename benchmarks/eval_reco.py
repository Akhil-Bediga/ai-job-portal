"""Evaluate the recommender against a naive keyword baseline.

Ground truth: a job is relevant to a resume if they share a job family (the
family tags are baked into the synthetic data and never shown to either model).
For every resume we rank all postings with both models and score the rankings
with precision@k, NDCG@k, and MRR, then report the engine's relative uplift.

Runs straight off data/raw/ - no database required - so the headline number is
reproducible with one command.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import pipeline_cfg
from src.nlp.skills import extract_skills
from src.reco.baseline import KeywordBaseline
from src.reco.engine import JobDoc, Recommender

RAW = pipeline_cfg.data_dir / "raw"
K = 10


def precision_at_k(ranked: list[str], fam: dict, gold: str, k: int) -> float:
    return sum(1 for j in ranked[:k] if fam[j] == gold) / k


def ndcg_at_k(ranked: list[str], fam: dict, gold: str, k: int, n_relevant: int) -> float:
    dcg = sum((fam[j] == gold) / np.log2(i + 2) for i, j in enumerate(ranked[:k]))
    ideal = min(k, n_relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg else 0.0


def reciprocal_rank(ranked: list[str], fam: dict, gold: str) -> float:
    for i, j in enumerate(ranked):
        if fam[j] == gold:
            return 1 / (i + 1)
    return 0.0


def _uplift(baseline: float, engine: float) -> float:
    return round((engine - baseline) / baseline * 100, 1) if baseline else 0.0


def run() -> dict:
    jobs = pd.read_csv(RAW / "jobs.csv")
    texts = json.loads((RAW / "jobs_text.json").read_text())
    resumes = json.loads((RAW / "resumes.json").read_text())
    fam = dict(zip(jobs["job_id"], jobs["family"]))
    rel_counts = jobs["family"].value_counts().to_dict()

    docs = [JobDoc(jid, texts[jid], extract_skills(texts[jid])) for jid in jobs["job_id"]]
    print(f"fitting models on {len(docs):,} postings, evaluating {len(resumes):,} resumes")
    engine = Recommender().fit(docs)
    baseline = KeywordBaseline().fit(docs)

    acc = {m: {"baseline": [], "engine": []} for m in ("precision", "ndcg", "mrr")}
    for r in resumes:
        gold = r["family"]
        n_rel = rel_counts[gold]
        for name, model in (("baseline", baseline), ("engine", engine)):
            ranked = [jid for jid, _ in model.rank_all(r["text"])]
            acc["precision"][name].append(precision_at_k(ranked, fam, gold, K))
            acc["ndcg"][name].append(ndcg_at_k(ranked, fam, gold, K, n_rel))
            acc["mrr"][name].append(reciprocal_rank(ranked, fam, gold))

    metrics = {}
    label = {"precision": f"precision@{K}", "ndcg": f"ndcg@{K}", "mrr": "mrr"}
    for m, sides in acc.items():
        b, e = float(np.mean(sides["baseline"])), float(np.mean(sides["engine"]))
        metrics[label[m]] = {"baseline": round(b, 4), "engine": round(e, 4), "uplift_pct": _uplift(b, e)}

    results = {
        "dataset": {"jobs": len(jobs), "resumes": len(resumes), "families": jobs["family"].nunique()},
        "k": K,
        "metrics": metrics,
    }

    print(f"\n{'metric':<14}{'baseline':>10}{'engine':>10}{'uplift':>10}")
    print("-" * 44)
    for name, v in metrics.items():
        print(f"{name:<14}{v['baseline']:>10.3f}{v['engine']:>10.3f}{v['uplift_pct']:>9.1f}%")

    out = pipeline_cfg.data_dir.parent / "benchmarks" / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return results


if __name__ == "__main__":
    run()
