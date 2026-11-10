"""Generate synthetic job postings and resumes.

Real job boards rate-limit hard and their terms usually forbid scraping, so I
generate the data instead. Every posting and resume is tagged with a job
"family" (Data Engineer, Frontend, etc.). That label is only used to score the
recommender in benchmarks/eval_reco.py - the model never sees it.

Each description is a generic boilerplate intro/outro (shared across every
family) wrapped around a few family-specific responsibility lines. The shared
boilerplate is intentional: it's what makes a naive keyword match struggle and
TF-IDF earn its keep.

Outputs (data/raw/):
  jobs.csv        structured columns
  jobs_text.json  {job_id: full description}
  resumes.json    [{resume_id, family, text}]
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from src.config import pipeline_cfg

fake = Faker("en_US")
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# family -> titles + the skills that actually define the role
FAMILIES: dict[str, dict[str, list[str]]] = {
    "Data Engineer": {
        "titles": ["Data Engineer", "Big Data Engineer", "ETL Developer", "Data Platform Engineer"],
        "skills": ["python", "sql", "apache spark", "apache airflow", "apache kafka",
                   "etl", "data warehouse", "snowflake", "dbt", "hadoop", "aws"],
    },
    "Data Scientist": {
        "titles": ["Data Scientist", "Applied Scientist", "Research Scientist"],
        "skills": ["python", "pandas", "numpy", "scikit learn", "machine learning",
                   "statistics", "a/b testing", "sql", "matplotlib", "deep learning"],
    },
    "ML Engineer": {
        "titles": ["Machine Learning Engineer", "ML Engineer", "MLOps Engineer"],
        "skills": ["python", "pytorch", "tensorflow", "machine learning", "deep learning",
                   "scikit learn", "feature engineering", "aws", "docker", "kubernetes"],
    },
    "Backend Engineer": {
        "titles": ["Backend Engineer", "Backend Developer", "Software Engineer (Backend)"],
        "skills": ["java", "spring boot", "rest api", "microservices", "postgresql",
                   "docker", "kubernetes", "kafka", "redis", "graphql"],
    },
    "Frontend Engineer": {
        "titles": ["Frontend Engineer", "Frontend Developer", "UI Engineer"],
        "skills": ["javascript", "typescript", "react", "redux", "html", "css",
                   "webpack", "responsive design", "next js", "accessibility"],
    },
    "DevOps Engineer": {
        "titles": ["DevOps Engineer", "Site Reliability Engineer", "Platform Engineer"],
        "skills": ["aws", "docker", "kubernetes", "terraform", "ci cd", "jenkins",
                   "linux", "bash", "github actions", "git"],
    },
    "Data Analyst": {
        "titles": ["Data Analyst", "Business Intelligence Analyst", "Analytics Engineer"],
        "skills": ["sql", "excel", "tableau", "power bi", "python", "statistics",
                   "data visualization", "dashboards", "pandas", "a/b testing"],
    },
    "Mobile Engineer": {
        "titles": ["Mobile Engineer", "iOS Engineer", "Android Engineer"],
        "skills": ["kotlin", "swift", "android", "ios", "react native", "flutter",
                   "java", "rest api"],
    },
}

# families that overlap in the real world - hybrid resumes blend two from a cluster
CLUSTERS = [
    ["Data Engineer", "Data Scientist", "ML Engineer", "Data Analyst"],
    ["Backend Engineer", "Frontend Engineer", "DevOps Engineer", "Mobile Engineer"],
]
HYBRID_PROB = 0.35  # share of resumes that mix a second, related family

SENIORITY = {"JUNIOR": 0.78, "MID": 1.0, "SENIOR": 1.30, "LEAD": 1.60}
SENIORITY_PREFIX = {"JUNIOR": "Junior ", "MID": "", "SENIOR": "Senior ", "LEAD": "Lead "}
EMPLOYMENT = ["FULL_TIME", "FULL_TIME", "FULL_TIME", "CONTRACT", "PART_TIME"]

# rough US base comp per family (MID level), scaled by seniority later
FAMILY_BASE = {
    "Data Engineer": 118_000, "Data Scientist": 122_000, "ML Engineer": 132_000,
    "Backend Engineer": 116_000, "Frontend Engineer": 108_000, "DevOps Engineer": 124_000,
    "Data Analyst": 86_000, "Mobile Engineer": 112_000,
}

# shared across every posting - this is the noise TF-IDF has to see through
INTRO = [
    "We are a fast growing company on a mission to build products our customers love.",
    "Join a collaborative team that values ownership, curiosity, and continuous learning.",
    "Our team partners across the organization to deliver impact at scale.",
    "We believe great software is built by empowered, cross functional teams.",
]
OUTRO = [
    "We offer competitive compensation, generous paid time off, and comprehensive health benefits.",
    "This role can be remote or hybrid depending on your location and preference.",
    "We are an equal opportunity employer and celebrate diversity on our teams.",
    "Apply today and help us shape the future of our platform.",
]
# generic professional fluff that shows up in BOTH postings and resumes. It
# carries no role signal, but a raw keyword match treats it as if it does - this
# varied boilerplate is the main reason the baseline ranks worse than the TF-IDF
# engine (high document frequency -> IDF down-weights it, raw term counts don't).
FILLER = [
    "Results driven professional passionate about solving hard problems and delivering value.",
    "Strong communication skills and a track record of collaborating with cross functional teams.",
    "Comfortable working in a fast paced agile environment and owning projects end to end.",
    "Committed to writing clean, well tested, maintainable, and well documented code.",
    "Excited about mentoring teammates and continuously learning new tools and best practices.",
    "Proven ability to translate ambiguous business requirements into reliable technical solutions.",
    "Detail oriented with strong analytical, organizational, and problem solving abilities.",
    "Adaptable team player who thrives on ownership, impact, and a culture of continuous improvement.",
    "We move fast, value pragmatism, and care deeply about the experience we ship to customers.",
    "Our company is backed by leading investors and growing quickly across multiple markets.",
    "You will work closely with product managers, designers, and other engineers every day.",
    "We offer competitive pay, equity, generous time off, and comprehensive medical coverage.",
    "Flexible hybrid schedule with offices in several cities and strong support for remote work.",
    "We invest in your growth with a learning budget, conferences, and dedicated mentorship.",
    "Our interview process is structured, transparent, and respectful of your time.",
    "We are committed to building an inclusive workplace and are an equal opportunity employer.",
    "You bring curiosity, a bias for action, and a willingness to learn from feedback.",
    "Excellent written and verbal communication, with the ability to explain tradeoffs clearly.",
    "Self starter who can scope ambiguous problems and drive them to completion with minimal oversight.",
    "Track record of shipping high quality work on schedule in collaborative environments.",
    "Comfortable balancing long term technical health with short term delivery pressure.",
    "You enjoy reviewing code, writing docs, and raising the bar for the whole team.",
    "Passionate about building reliable systems that are a pleasure for teammates to work with.",
    "We believe small empowered teams with clear ownership produce the best outcomes.",
    "Thrives on feedback, retrospectives, and a culture of continuous, measurable improvement.",
    "Strong sense of urgency balanced with sound judgment and attention to detail.",
    "You care about the customer, measure what matters, and iterate quickly toward impact.",
    "Generous parental leave, wellness stipend, and a budget for your home office setup.",
]
N_RESP = 3        # role-specific responsibility lines per posting
JOB_FILLER = 5    # generic filler lines per posting
RESUME_FILLER = 4 # generic filler lines per resume
# generic responsibility scaffolding; the {skill} slots carry the real signal
RESP_TEMPLATES = [
    "You will design, build, and maintain solutions using {a} and {b}.",
    "Collaborate with stakeholders to deliver features powered by {a}.",
    "Own the end to end lifecycle of systems built on {a} and {b}.",
    "Improve reliability and performance across our {a} stack.",
    "Partner with the team to scale our use of {a} and {b}.",
]
REQ_TEMPLATES = [
    "Strong hands on experience with {a}.",
    "Proven track record working with {a} and {b} in production.",
    "Solid understanding of {a}; exposure to {b} is a plus.",
]


def _sentence(template: str, skills: list[str]) -> str:
    a, b = random.sample(skills, 2) if len(skills) >= 2 else (skills[0], skills[0])
    return template.format(a=a, b=b)


def _description(family: str) -> str:
    core = FAMILIES[family]["skills"]
    # skills borrowed from other families as realistic noise (real postings are
    # rarely "pure") - this is what stops the engine from being trivially perfect
    other = [s for f, d in FAMILIES.items() if f != family for s in d["skills"]]
    noise = random.sample(other, k=2)

    parts: list[str] = [random.choice(INTRO)]
    parts += random.sample(FILLER, k=JOB_FILLER)
    parts += [_sentence(t, core) for t in random.sample(RESP_TEMPLATES, k=N_RESP)]

    req_skills = core + noise
    parts.append("Requirements:")
    parts += ["- " + _sentence(t, req_skills) for t in random.sample(REQ_TEMPLATES, k=2)]
    parts.append("Nice to have: " + ", ".join(random.sample(req_skills, k=min(4, len(req_skills)))) + ".")
    parts.append(random.choice(OUTRO))
    return "\n".join(parts)


def _salary(family: str, seniority: str) -> tuple[int, int]:
    base = FAMILY_BASE[family] * SENIORITY[seniority]
    avg = base * random.uniform(0.92, 1.08)
    lo = int(round(avg * 0.9, -3))
    hi = int(round(avg * 1.12, -3))
    return lo, hi


def gen_jobs(n: int) -> tuple[pd.DataFrame, dict[str, str]]:
    families = list(FAMILIES)
    rows = []
    texts: dict[str, str] = {}
    today = date.today()
    for _ in range(n):
        family = random.choice(families)
        seniority = random.choices(list(SENIORITY), weights=[0.2, 0.45, 0.25, 0.1])[0]
        title = SENIORITY_PREFIX[seniority] + random.choice(FAMILIES[family]["titles"])
        job_id = str(uuid.uuid4())
        lo, hi = _salary(family, seniority)
        rows.append({
            "job_id": job_id,
            "title": title,
            "company": fake.company(),
            "family": family,
            "city": fake.city(),
            "state": fake.state_abbr(),
            "remote": int(random.random() < 0.35),
            "employment_type": random.choice(EMPLOYMENT),
            "seniority": seniority,
            "salary_min": lo,
            "salary_max": hi,
            "posted_date": today - timedelta(days=random.randint(0, 60)),
        })
        texts[job_id] = _description(family)
    return pd.DataFrame(rows), texts


def _related_family(family: str) -> str:
    cluster = next(c for c in CLUSTERS if family in c)
    return random.choice([f for f in cluster if f != family])


def _resume(family: str, secondary: str | None = None) -> str:
    core = FAMILIES[family]["skills"]
    primary = FAMILIES[family]["titles"][0]
    years = random.randint(2, 9)
    lines = [
        f"{primary} with {years} years of experience building and shipping production systems.",
    ]
    lines += random.sample(FILLER, k=RESUME_FILLER)
    lines.append("Experience:")
    lines += ["- " + _sentence(t, core) for t in random.sample(RESP_TEMPLATES, k=N_RESP)]
    if secondary:
        # a hybrid background - a real foot in a related role, not just a mention
        sec_skills = FAMILIES[secondary]["skills"]
        lines += ["- " + _sentence(t, sec_skills) for t in random.sample(RESP_TEMPLATES, k=2)]

    # a real resume rarely lists every core skill and usually picks up adjacent ones
    listed = random.sample(core, k=max(4, len(core) - 2))
    if secondary:
        listed += random.sample(FAMILIES[secondary]["skills"], k=4)
    else:
        other = [s for f, d in FAMILIES.items() if f != family for s in d["skills"]]
        listed += random.sample(other, k=2)
    random.shuffle(listed)
    lines.append("Skills: " + ", ".join(dict.fromkeys(listed)) + ".")
    return "\n".join(lines)


def gen_resumes(n: int) -> list[dict]:
    families = list(FAMILIES)
    out = []
    for i in range(n):
        family = families[i % len(families)]  # even spread across families
        secondary = _related_family(family) if random.random() < HYBRID_PROB else None
        out.append({
            "resume_id": f"r{i:04d}",
            "family": family,
            "text": _resume(family, secondary),
        })
    random.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1200, help="Number of job postings.")
    parser.add_argument("--resumes", type=int, default=240, help="Number of resumes.")
    args = parser.parse_args()

    raw_dir = pipeline_cfg.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"generating {args.jobs:,} job postings")
    jobs, texts = gen_jobs(args.jobs)
    jobs.to_csv(raw_dir / "jobs.csv", index=False)
    (raw_dir / "jobs_text.json").write_text(json.dumps(texts))

    print(f"generating {args.resumes:,} resumes")
    resumes = gen_resumes(args.resumes)
    (raw_dir / "resumes.json").write_text(json.dumps(resumes, indent=2))

    by_family = jobs["family"].value_counts().to_dict()
    print(f"wrote raw data to {raw_dir}")
    for fam, cnt in sorted(by_family.items()):
        print(f"    {fam:<20} {cnt:>5} jobs")


if __name__ == "__main__":
    main()
