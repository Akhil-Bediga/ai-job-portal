"""Unit tests for the recommendation engine and baseline."""
from src.reco.baseline import KeywordBaseline
from src.reco.engine import JobDoc, Recommender, _jaccard

JOBS = [
    JobDoc("de", "python sql apache spark airflow etl data warehouse snowflake",
           ["python", "sql", "apache spark", "etl", "snowflake"]),
    JobDoc("fe", "javascript typescript react redux html css responsive design",
           ["javascript", "typescript", "react", "css"]),
    JobDoc("devops", "aws docker kubernetes terraform ci cd jenkins linux bash",
           ["aws", "docker", "kubernetes", "terraform"]),
]


def test_recommend_returns_top_n_sorted_desc():
    reco = Recommender().fit(JOBS)
    matches = reco.recommend("python sql etl snowflake spark", top_n=2)
    assert len(matches) == 2
    assert matches[0].score >= matches[1].score


def test_engine_ranks_matching_family_first():
    reco = Recommender().fit(JOBS)
    top = reco.recommend("I work with python, sql, apache spark and snowflake", top_n=1)[0]
    assert top.job_id == "de"
    assert top.matched_skills  # surfaced the overlapping skills


def test_missing_skills_are_reported():
    reco = Recommender().fit(JOBS)
    top = reco.recommend("python sql", top_n=1)[0]
    # the DE posting wants more than the resume lists
    assert "apache spark" in top.missing_skills or "snowflake" in top.missing_skills


def test_rank_all_covers_every_job():
    reco = Recommender().fit(JOBS)
    ranked = reco.rank_all("react typescript css")
    assert {jid for jid, _ in ranked} == {"de", "fe", "devops"}


def test_baseline_ranks_and_returns_top_n():
    base = KeywordBaseline().fit(JOBS)
    out = base.recommend("react typescript css redux html", top_n=1)
    assert out[0][0] == "fe"


def test_jaccard_bounds():
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a"}, {"b"}) == 0.0
    assert _jaccard(set(), set()) == 0.0
